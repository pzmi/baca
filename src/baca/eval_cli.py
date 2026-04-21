"""Evaluation CLI: load a checkpoint, play N games, print winrate + avg floor."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
from gymnasium import Env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

from baca.env import UsiecCepraEnv
from baca.rpc_client import RpcClient


def _mask_fn(env: Env[Any, Any]) -> np.ndarray:
    return cast(UsiecCepraEnv, env).action_masks()


class EpisodeCollector:
    """Per-step callback for ``evaluate_policy`` that records terminal info.

    ``evaluate_policy`` passes ``locals()`` which exposes ``done`` and ``info``
    from the current vectorized step. On episode end we harvest ``outcome`` and
    ``summary.floorReached`` that :class:`UsiecCepraEnv` attaches in ``info``.
    """

    def __init__(self, total: int, report_every: int = 10) -> None:
        self.floors: list[int] = []
        self.wins: list[int] = []
        self.total = total
        self.report_every = max(1, report_every)
        self._start = time.monotonic()

    def __call__(self, locals_dict: dict[str, Any], _globals_dict: dict[str, Any]) -> None:
        if not locals_dict.get("done"):
            return
        info = locals_dict.get("info") or {}
        summary = info.get("summary") or {}
        self.floors.append(int(summary.get("floorReached", 0)))
        self.wins.append(1 if info.get("outcome") == "player_win" else 0)
        n = len(self.wins)
        if n % self.report_every == 0 or n == self.total:
            elapsed = time.monotonic() - self._start
            wr = sum(self.wins) / n
            avg_fl = statistics.fmean(self.floors)
            print(  # noqa: T201 — CLI progress
                f"[eval] {n}/{self.total}  winrate={wr:.3f}  avg_floor={avg_fl:.2f}  "
                f"elapsed={elapsed:.1f}s",
                flush=True,
                file=sys.stderr,
            )


def _aggregate(episodes: int, floors: list[int], wins: list[int]) -> dict[str, float]:
    finished = len(wins)
    return {
        "episodes": float(episodes),
        "winrate": (sum(wins) / finished) if finished else 0.0,
        "avg_floor": statistics.fmean(floors) if floors else 0.0,
        "max_floor": float(max(floors)) if floors else 0.0,
    }


def evaluate(
    checkpoint: Path,
    episodes: int,
    character_id: str,
    difficulty: str,
    base_seed: int | None,
    engine_dir: Path | None,
) -> dict[str, float]:
    collector = EpisodeCollector(total=episodes)
    rpc = RpcClient(engine_dir=engine_dir)
    try:
        raw_env = UsiecCepraEnv(
            rpc,
            character_id=character_id,
            difficulty=difficulty,
            base_seed=base_seed,
        )
        env: Monitor[Any, Any] = Monitor(ActionMasker(raw_env, _mask_fn))
        model = MaskablePPO.load(str(checkpoint), env=env)
        evaluate_policy(
            model,
            env,
            n_eval_episodes=episodes,
            deterministic=True,
            use_masking=True,
            callback=collector,
        )
        env.close()
    finally:
        rpc.close()
    return _aggregate(episodes, collector.floors, collector.wins)


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
