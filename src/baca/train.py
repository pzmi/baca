"""MaskablePPO training entrypoint for BACA v0.

Spawns multiple Usiec Cepra engine subprocesses (one per worker, plus a
separate eval subprocess), wraps each in a gymnasium env, and trains a
masked-action PPO policy with periodic evaluation and checkpointing.

Vectorization uses :class:`SubprocVecEnv` so each worker owns its own
``RpcClient`` (the client is not picklable across processes). MaskablePPO
reads per-env masks via ``env.env_method("action_masks")`` on the VecEnv,
so the explicit :class:`ActionMasker` wrapper is unneeded on the training
path.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from baca.env import UsiecCepraEnv
from baca.policy import MaskableBacaPolicy
from baca.rpc_client import RpcClient


@dataclass(frozen=True)
class TrainConfig:
    total_timesteps: int
    character_id: str
    difficulty: str
    base_seed: int | None
    checkpoint_dir: Path
    engine_dir: Path | None
    learning_rate: float
    n_steps: int
    batch_size: int
    ent_coef: float = 0.01
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    lr_final: float = 1e-4
    eval_freq: int = 10_000
    eval_episodes: int = 50
    checkpoint_freq: int = 20_000
    seed: int = 42
    n_envs: int = 8
    reward_shape: str = "floor"
    card_embed_dim: int = 32
    features_dim: int = 64
    bc_init: Path | None = None
    initial_lr_scale: float = 1.0
    warmup_frac: float = 0.1
    reset_num_timesteps: bool = True


def _linear_schedule(start: float, end: float) -> Callable[[float], float]:
    """Anneal ``start`` → ``end`` as SB3 progress_remaining goes 1.0 → 0.0."""

    def schedule(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining

    return schedule


def _warmup_schedule(
    lr_start: float,
    lr_final: float,
    warmup_frac: float,
    initial_scale: float,
) -> Callable[[float], float]:
    """Linear LR schedule with an early warmup window scaled by ``initial_scale``.

    Within the first ``warmup_frac`` of training (progress_remaining in
    ``[1 - warmup_frac, 1.0]``), the base schedule is multiplied by a factor
    that interpolates from ``initial_scale`` at the start to ``1.0`` at the end
    of warmup. After warmup the base schedule applies verbatim. Degenerates to
    ``_linear_schedule`` when ``initial_scale == 1.0`` or ``warmup_frac == 0``.
    """
    base = _linear_schedule(lr_start, lr_final)
    if initial_scale == 1.0 or warmup_frac <= 0.0:
        return base

    warmup_cutoff = 1.0 - warmup_frac

    def schedule(progress_remaining: float) -> float:
        base_lr = base(progress_remaining)
        if progress_remaining <= warmup_cutoff:
            return base_lr
        # t = 0 at start of training (progress_remaining == 1.0);
        # t = 1 at end of warmup (progress_remaining == warmup_cutoff).
        t = (1.0 - progress_remaining) / warmup_frac
        scale = initial_scale + (1.0 - initial_scale) * t
        return scale * base_lr

    return schedule


def make_env(
    rank: int,
    *,
    character_id: str,
    difficulty: str,
    base_seed: int | None,
    engine_dir: Path | None,
    reward_shape: str = "none",
) -> Callable[[], gymnasium.Env[Any, Any]]:
    """Build a picklable thunk that a worker calls to construct its own env.

    Each invocation of the returned callable runs inside the worker process
    and constructs a dedicated :class:`RpcClient` (spawning one Node engine
    subprocess per worker), keeping the 16-run-per-process cap well below
    saturation for typical ``n_envs`` values.
    """

    worker_seed = None if base_seed is None else base_seed + rank * 1_000_000
    # Cast once, here, so the literal survives the pickle boundary to subproc
    # workers and the worker-side import from baca.env typechecks.
    from baca.env import RewardShape  # imported lazily to avoid cycle surprises

    shape_literal: RewardShape = "floor" if reward_shape == "floor" else "none"

    def _thunk() -> gymnasium.Env[Any, Any]:
        rpc = RpcClient(engine_dir=engine_dir)
        env = UsiecCepraEnv(
            rpc,
            character_id=character_id,
            difficulty=difficulty,
            base_seed=worker_seed,
            reward_shape=shape_literal,
        )
        return Monitor(env)

    return _thunk


def _build_train_vec_env(cfg: TrainConfig) -> VecEnv:
    train_seed = cfg.base_seed if cfg.base_seed is not None else cfg.seed
    thunks = [
        make_env(
            rank=i,
            character_id=cfg.character_id,
            difficulty=cfg.difficulty,
            base_seed=train_seed,
            engine_dir=cfg.engine_dir,
            reward_shape=cfg.reward_shape,
        )
        for i in range(cfg.n_envs)
    ]
    if cfg.n_envs == 1:
        return DummyVecEnv(thunks)
    return SubprocVecEnv(thunks, start_method="spawn")


def _build_eval_vec_env(cfg: TrainConfig) -> VecEnv:
    train_seed = cfg.base_seed if cfg.base_seed is not None else cfg.seed
    eval_thunk = make_env(
        rank=0,
        character_id=cfg.character_id,
        difficulty=cfg.difficulty,
        base_seed=train_seed + 10_000,
        engine_dir=cfg.engine_dir,
        reward_shape=cfg.reward_shape,
    )
    return DummyVecEnv([eval_thunk])


def _load_bc_model(cfg: TrainConfig, vec_env: VecEnv) -> MaskablePPO:
    """Load a BC checkpoint as MaskablePPO and overwrite transient hyperparams.

    ``MaskablePPO.load`` restores every scalar attribute from the zip's pickled
    ``data`` dict. The BC checkpoint was saved via a transient MaskablePPO with
    ``n_steps=8``, ``batch_size=8``, ``n_epochs=1``, ``ent_coef=0.0`` — none of
    which the caller wants during fine-tuning. We override via ``custom_objects``
    for callables (LR + clip-range schedules) and reassign scalar attributes
    post-load. ``n_steps`` changes also require a fresh rollout buffer since
    SB3 materializes it eagerly at load time.
    """
    lr_schedule = _warmup_schedule(
        cfg.learning_rate, cfg.lr_final, cfg.warmup_frac, cfg.initial_lr_scale
    )
    clip_range_schedule = _linear_schedule(cfg.clip_range, cfg.clip_range)
    model = MaskablePPO.load(
        str(cfg.bc_init),
        env=vec_env,
        custom_objects={
            "learning_rate": lr_schedule,
            "lr_schedule": lr_schedule,
            "clip_range": clip_range_schedule,
        },
    )
    model.ent_coef = cfg.ent_coef
    model.n_epochs = cfg.n_epochs
    model.n_steps = cfg.n_steps
    model.batch_size = cfg.batch_size
    model.gamma = cfg.gamma
    model.gae_lambda = cfg.gae_lambda
    # Rebuild the rollout buffer so n_steps / gamma / gae_lambda actually take
    # effect; SB3 instantiates it inside _setup_model() based on the saved
    # scalars and we just overwrote those.
    buffer_cls = type(model.rollout_buffer)
    model.rollout_buffer = buffer_cls(
        cfg.n_steps,
        model.observation_space,
        model.action_space,
        model.device,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        n_envs=vec_env.num_envs,
    )
    return model


def train(cfg: TrainConfig) -> Path:
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_env = _build_train_vec_env(cfg)
    eval_env: VecEnv | None = None
    try:
        eval_env = _build_eval_vec_env(cfg)

        policy_kwargs: dict[str, Any] = {
            "card_embed_dim": cfg.card_embed_dim,
            "features_dim": cfg.features_dim,
        }
        if cfg.bc_init is not None:
            model = _load_bc_model(cfg, train_env)
        else:
            model = MaskablePPO(
                MaskableBacaPolicy,
                train_env,
                learning_rate=_warmup_schedule(
                    cfg.learning_rate, cfg.lr_final, cfg.warmup_frac, cfg.initial_lr_scale
                ),
                n_steps=cfg.n_steps,
                batch_size=cfg.batch_size,
                n_epochs=cfg.n_epochs,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
                clip_range=cfg.clip_range,
                ent_coef=cfg.ent_coef,
                seed=cfg.seed,
                verbose=1,
                tensorboard_log=str(cfg.checkpoint_dir / "tensorboard"),
                policy_kwargs=policy_kwargs,
            )

        eval_cb = MaskableEvalCallback(
            eval_env,
            best_model_save_path=str(cfg.checkpoint_dir / "best"),
            log_path=str(cfg.checkpoint_dir / "eval"),
            eval_freq=max(cfg.eval_freq // cfg.n_envs, 1),
            n_eval_episodes=cfg.eval_episodes,
            deterministic=True,
            render=False,
        )
        ckpt_cb = CheckpointCallback(
            save_freq=max(cfg.checkpoint_freq // cfg.n_envs, 1),
            save_path=str(cfg.checkpoint_dir / "ckpts"),
            name_prefix="baca",
        )
        callback = CallbackList([eval_cb, ckpt_cb])

        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=callback,
            reset_num_timesteps=cfg.reset_num_timesteps,
        )
        out_path = cfg.checkpoint_dir / "baca-latest.zip"
        model.save(str(out_path))
        return out_path
    finally:
        train_env.close()
        if eval_env is not None:
            eval_env.close()


def _parse_args(argv: list[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(prog="baca-train")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--character-id", type=str, default="jedrek")
    parser.add_argument("--difficulty", type=str, default="normal", choices=["normal", "hard"])
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--engine-dir", type=Path, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--lr-final", type=float, default=1e-4)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--checkpoint-freq", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Parallel training envs. Each worker owns one RpcClient (and one Node subprocess).",
    )
    parser.add_argument(
        "--reward-shape",
        type=str,
        default="floor",
        choices=["none", "floor"],
        help="Terminal reward shaping. 'floor' = 0.1*(floor/15) + 0.9*is_win; 'none' = v0 pure win signal.",
    )
    parser.add_argument("--card-embed-dim", type=int, default=32)
    parser.add_argument("--features-dim", type=int, default=64)
    parser.add_argument(
        "--bc-init",
        type=Path,
        default=None,
        help="Path to a MaskablePPO-compatible BC checkpoint to warm-start from.",
    )
    parser.add_argument(
        "--initial-lr-scale",
        type=float,
        default=1.0,
        help="Scale applied to the LR schedule during the first --warmup-frac of training.",
    )
    parser.add_argument(
        "--warmup-frac",
        type=float,
        default=0.1,
        help="Fraction of total_timesteps over which the initial-lr-scale interpolates to 1.0.",
    )
    parser.add_argument(
        "--reset-num-timesteps",
        dest="reset_num_timesteps",
        action="store_true",
        help="Reset model.num_timesteps at .learn() start (SB3 default).",
    )
    parser.add_argument(
        "--no-reset-num-timesteps",
        dest="reset_num_timesteps",
        action="store_false",
        help="Preserve model.num_timesteps across .learn() calls.",
    )
    parser.set_defaults(reset_num_timesteps=True)
    args = parser.parse_args(argv)
    return TrainConfig(
        total_timesteps=args.total_timesteps,
        character_id=args.character_id,
        difficulty=args.difficulty,
        base_seed=args.base_seed,
        checkpoint_dir=args.checkpoint_dir,
        engine_dir=args.engine_dir,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        lr_final=args.lr_final,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        checkpoint_freq=args.checkpoint_freq,
        seed=args.seed,
        n_envs=args.n_envs,
        reward_shape=args.reward_shape,
        card_embed_dim=args.card_embed_dim,
        features_dim=args.features_dim,
        bc_init=args.bc_init,
        initial_lr_scale=args.initial_lr_scale,
        warmup_frac=args.warmup_frac,
        reset_num_timesteps=args.reset_num_timesteps,
    )


def main(argv: list[str] | None = None) -> None:
    cfg = _parse_args(argv)
    out = train(cfg)
    print(f"Saved checkpoint to {out}")  # noqa: T201 — CLI output


if __name__ == "__main__":
    main()
