"""Evaluation CLI: load a checkpoint, play N games, print winrate + avg floor."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from baca.env import UsiecCepraEnv
from baca.rpc_client import RpcClient


def _mask_fn(env: UsiecCepraEnv) -> "np.ndarray":  # type: ignore[name-defined]
    return env.action_masks()


def evaluate(
    checkpoint: Path,
    episodes: int,
    character_id: str,
    difficulty: str,
    base_seed: int | None,
    engine_dir: Path | None,
) -> dict[str, float]:
    rpc = RpcClient(engine_dir=engine_dir)
    wins = 0
    floors: list[int] = []
    try:
        raw_env = UsiecCepraEnv(
            rpc,
            character_id=character_id,
            difficulty=difficulty,
            base_seed=base_seed,
        )
        env = ActionMasker(raw_env, _mask_fn)
        model = MaskablePPO.load(str(checkpoint), env=env)
        for _ in range(episodes):
            obs, _info = env.reset()
            done = False
            while not done:
                mask = raw_env.action_masks()
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                obs, _reward, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated
            summary = (info or {}).get("summary") or {}
            if (info or {}).get("outcome") == "player_win":
                wins += 1
            floors.append(int(summary.get("floorReached", 0)))
        env.close()
    finally:
        rpc.close()
    return {
        "episodes": float(episodes),
        "winrate": wins / max(episodes, 1),
        "avg_floor": statistics.fmean(floors) if floors else 0.0,
        "max_floor": float(max(floors)) if floors else 0.0,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="baca-eval")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--character-id", type=str, default="jedrek")
    parser.add_argument("--difficulty", type=str, default="normal", choices=["normal", "hard"])
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--engine-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    metrics = evaluate(
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        character_id=args.character_id,
        difficulty=args.difficulty,
        base_seed=args.base_seed,
        engine_dir=args.engine_dir,
    )
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")  # noqa: T201 — CLI output


if __name__ == "__main__":
    main()
