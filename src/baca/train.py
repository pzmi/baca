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
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    lr_final: float = 1e-4
    eval_freq: int = 10_000
    eval_episodes: int = 50
    checkpoint_freq: int = 20_000
    seed: int = 42
    n_envs: int = 8


def _linear_schedule(start: float, end: float) -> Callable[[float], float]:
    """Anneal ``start`` → ``end`` as SB3 progress_remaining goes 1.0 → 0.0."""

    def schedule(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining

    return schedule


def make_env(
    rank: int,
    *,
    character_id: str,
    difficulty: str,
    base_seed: int | None,
    engine_dir: Path | None,
) -> Callable[[], gymnasium.Env[Any, Any]]:
    """Build a picklable thunk that a worker calls to construct its own env.

    Each invocation of the returned callable runs inside the worker process
    and constructs a dedicated :class:`RpcClient` (spawning one Node engine
    subprocess per worker), keeping the 16-run-per-process cap well below
    saturation for typical ``n_envs`` values.
    """

    worker_seed = None if base_seed is None else base_seed + rank * 1_000_000

    def _thunk() -> gymnasium.Env[Any, Any]:
        rpc = RpcClient(engine_dir=engine_dir)
        env = UsiecCepraEnv(
            rpc,
            character_id=character_id,
            difficulty=difficulty,
            base_seed=worker_seed,
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
    )
    return DummyVecEnv([eval_thunk])


def train(cfg: TrainConfig) -> Path:
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_env = _build_train_vec_env(cfg)
    eval_env: VecEnv | None = None
    try:
        eval_env = _build_eval_vec_env(cfg)

        model = MaskablePPO(
            "MultiInputPolicy",
            train_env,
            learning_rate=_linear_schedule(cfg.learning_rate, cfg.lr_final),
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

        model.learn(total_timesteps=cfg.total_timesteps, callback=callback)
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
    parser.add_argument("--gamma", type=float, default=0.995)
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
    )


def main(argv: list[str] | None = None) -> None:
    cfg = _parse_args(argv)
    out = train(cfg)
    print(f"Saved checkpoint to {out}")  # noqa: T201 — CLI output


if __name__ == "__main__":
    main()
