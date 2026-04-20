"""MaskablePPO training entrypoint for BACA v0.

Spawns a Usiec Cepra engine subprocess, wraps it in a gymnasium env, and trains
a masked-action PPO policy. A single env instance is used — vectorization will
come in a later phase once the pipeline is validated end-to-end.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

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


def _mask_fn(env: UsiecCepraEnv) -> "np.ndarray":  # type: ignore[name-defined]
    return env.action_masks()


def train(cfg: TrainConfig) -> Path:
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rpc = RpcClient(engine_dir=cfg.engine_dir)
    try:
        raw_env = UsiecCepraEnv(
            rpc,
            character_id=cfg.character_id,
            difficulty=cfg.difficulty,
            base_seed=cfg.base_seed,
        )
        env = Monitor(ActionMasker(raw_env, _mask_fn))
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            verbose=1,
            tensorboard_log=str(cfg.checkpoint_dir / "tensorboard"),
        )
        model.learn(total_timesteps=cfg.total_timesteps)
        out_path = cfg.checkpoint_dir / "baca-latest.zip"
        model.save(str(out_path))
        return out_path
    finally:
        rpc.close()


def _parse_args(argv: list[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(prog="baca-train")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--character-id", type=str, default="jedrek")
    parser.add_argument("--difficulty", type=str, default="normal", choices=["normal", "hard"])
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints")
    )
    parser.add_argument("--engine-dir", type=Path, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
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
    )


def main(argv: list[str] | None = None) -> None:
    cfg = _parse_args(argv)
    out = train(cfg)
    print(f"Saved checkpoint to {out}")  # noqa: T201 — CLI output


if __name__ == "__main__":
    main()
