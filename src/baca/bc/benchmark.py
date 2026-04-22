"""Step 0 throughput measurement for iter-3 BC data collection.

One-shot script. Measures:
  * games/sec and avg decision steps per game when driving the engine
    directly via RpcClient + HeuristicBridge;
  * RPC roundtrip latency on ``engine.getObservation`` (p50/p95);
  * HeuristicBridge.decide latency (p50/p95).

Prints a fenced markdown block suitable for pasting into ``doc/findings.md``.
Does NOT auto-write to findings.md.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from baca.bc.heuristic_bridge import HeuristicBridge
from baca.rpc_client import RpcClient

RPC_PING_COUNT = 500
MAX_STEPS_PER_GAME = 5000


def _resolve_engine_dir() -> Path:
    override = os.environ.get("BACA_ENGINE_DIR")
    if override:
        return Path(override).resolve()
    return Path("../slay-the-ceper").resolve()


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[k]


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _action_index(action: Any, legal_actions: list[Any]) -> int:
    target = _canonical(action)
    for i, candidate in enumerate(legal_actions):
        if _canonical(candidate) == target:
            return i
    raise ValueError(f"heuristic action {action!r} not in legal list")


def _start_run(
    rpc: RpcClient, character_id: str, difficulty: str, seed: int
) -> tuple[str, dict[str, Any]]:
    create_params: dict[str, Any] = {
        "characterId": character_id,
        "difficulty": difficulty,
        "marynaEnabled": True,
        "rules": {"revealAllPiles": False, "observationMode": "agent"},
        "seed": seed,
    }
    run = rpc.call("engine.create", create_params)
    run_id = run["runId"]
    start = rpc.call("engine.startRun", {"runId": run_id})
    return run_id, start["observation"]


def main() -> int:
    parser = argparse.ArgumentParser(description="BACA iter-3 Step 0 benchmark")
    parser.add_argument("--games", type=int, default=20, help="Number of games to play")
    parser.add_argument("--base-seed", type=int, default=1, help="First seed to play")
    parser.add_argument("--character-id", type=str, default="jedrek")
    parser.add_argument("--difficulty", type=str, default="normal")
    args = parser.parse_args()

    engine_dir = _resolve_engine_dir()
    if not (engine_dir / "scripts" / "rpc-server.js").is_file():
        print(  # noqa: T201 — CLI output
            f"engine missing: no rpc-server.js under {engine_dir}", file=sys.stderr
        )
        return 2

    rpc = RpcClient(engine_dir=engine_dir)
    bridge = HeuristicBridge(engine_dir=engine_dir)

    try:
        game_steps: list[int] = []
        game_times: list[float] = []
        heuristic_ms: list[float] = []

        for i in range(args.games):
            seed = args.base_seed + i
            run_id, obs = _start_run(rpc, args.character_id, args.difficulty, seed)
            game_start = time.perf_counter()
            steps = 0
            while not bool(obs.get("done")) and steps < MAX_STEPS_PER_GAME:
                legal = obs.get("legalActions") or []
                if not legal:
                    break
                h_start = time.perf_counter()
                action_obj = bridge.decide(obs)
                heuristic_ms.append((time.perf_counter() - h_start) * 1000.0)
                idx = _action_index(action_obj, legal)
                result = rpc.call("engine.applyAction", {"runId": run_id, "action": legal[idx]})
                obs = result["observation"]
                steps += 1
            game_times.append(time.perf_counter() - game_start)
            game_steps.append(steps)
            with contextlib.suppress(Exception):
                rpc.call("engine.dispose", {"runId": run_id})

        # RPC micro-bench: one live run, RPC_PING_COUNT getObservation pings.
        run_id, _obs = _start_run(
            rpc, args.character_id, args.difficulty, args.base_seed + args.games
        )
        rpc_ms: list[float] = []
        for _ in range(RPC_PING_COUNT):
            start = time.perf_counter()
            rpc.call("engine.getObservation", {"runId": run_id})
            rpc_ms.append((time.perf_counter() - start) * 1000.0)
        with contextlib.suppress(Exception):
            rpc.call("engine.dispose", {"runId": run_id})

        avg_steps = statistics.fmean(game_steps) if game_steps else float("nan")
        total_time = sum(game_times) if game_times else 0.0
        games_per_sec = (len(game_times) / total_time) if total_time > 0 else float("nan")
        projected_50k_hours = (
            (50000 / games_per_sec) / 3600.0 if games_per_sec > 0 else float("nan")
        )

        block = [
            f"## Step 0 measurements ({datetime.now(UTC).date().isoformat()})",
            "```",
            f"avg_steps_per_game: {avg_steps:.1f}",
            f"p50 rpc_ms: {_percentile(rpc_ms, 50):.2f}",
            f"p95 rpc_ms: {_percentile(rpc_ms, 95):.2f}",
            f"p50 heuristic_ms: {_percentile(heuristic_ms, 50):.2f}",
            f"p95 heuristic_ms: {_percentile(heuristic_ms, 95):.2f}",
            f"games_per_sec: {games_per_sec:.2f}",
            f"projected_wall_clock_50k_games: {projected_50k_hours:.2f} h",
            "```",
        ]
        print("\n".join(block))  # noqa: T201 — CLI output
        return 0
    finally:
        bridge.close()
        rpc.close()


if __name__ == "__main__":
    sys.exit(main())
