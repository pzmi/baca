"""Dataset generator: drive one engine run with HeuristicBot, write NPZ-per-game.

One NPZ file per game, flat layout in the output directory (no sharding).
The sharding switch-over (to ``seeds/XX/...``) is tracked as R8 in the iter-3
plan; revisit if 50k+ files in one directory slows filesystem ops.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from baca.bc.heuristic_bridge import HeuristicBridge
from baca.encoder import MAX_ACTIONS, encode
from baca.rpc_client import RpcClient


class GameRecord(TypedDict, total=False):
    """One game's demonstration trace.

    ``encoded_obs`` is a dict of stacked numpy arrays, one row per decision
    step (axis 0 length = n_steps). ``action_indices`` is a 1-D int64 array of
    the same length giving the HeuristicBot-chosen index into
    ``observation.legalActions`` at each step.
    """

    encoded_obs: dict[str, np.ndarray]
    action_indices: np.ndarray
    n_steps: int
    seed: int
    character_id: str
    difficulty: str
    outcome: str
    floor_reached: int
    raw_observations: list[dict[str, Any]]


@dataclass
class _Trace:
    encoded_rows: dict[str, list[np.ndarray]] = field(default_factory=dict)
    action_indices: list[int] = field(default_factory=list)
    raw_observations: list[dict[str, Any]] = field(default_factory=list)

    def append(self, encoded: dict[str, np.ndarray], action_idx: int) -> None:
        for key, value in encoded.items():
            self.encoded_rows.setdefault(key, []).append(value)
        self.action_indices.append(action_idx)

    def stack(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        stacked: dict[str, np.ndarray] = {
            key: np.stack(rows, axis=0) for key, rows in self.encoded_rows.items()
        }
        actions = np.asarray(self.action_indices, dtype=np.int64)
        return stacked, actions


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def action_to_index(heuristic_action: dict[str, Any], legal_actions: list[dict[str, Any]]) -> int:
    """Structural equality lookup of ``heuristic_action`` in ``legal_actions``.

    Uses canonical JSON (sort_keys) for stable comparison. Raises ``ValueError``
    if no legal action matches.
    """
    target = _canonical(heuristic_action)
    for i, candidate in enumerate(legal_actions):
        if _canonical(candidate) == target:
            return i
    raise ValueError(f"heuristic action {heuristic_action!r} not in legal list")


def _create_run(
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
    run_id: str = run["runId"]
    start = rpc.call("engine.startRun", {"runId": run_id})
    first_obs: dict[str, Any] = start["observation"]
    return run_id, first_obs


def collect_one_game(
    rpc: RpcClient,
    bridge: HeuristicBridge,
    seed: int,
    *,
    character_id: str = "jedrek",
    difficulty: str = "normal",
    max_steps: int = 5000,
    return_raw: bool = False,
) -> GameRecord:
    """Play one HeuristicBot-driven game and return the encoded trace."""
    run_id, obs = _create_run(rpc, character_id, difficulty, seed)
    trace = _Trace()
    try:
        steps = 0
        while not bool(obs.get("done")) and steps < max_steps:
            legal = obs.get("legalActions") or []
            if not legal:
                break
            heuristic_action = bridge.decide(obs)
            idx = action_to_index(heuristic_action, legal)
            if idx >= MAX_ACTIONS:
                raise ValueError(
                    f"action index {idx} exceeds MAX_ACTIONS={MAX_ACTIONS}; legal list too long"
                )
            encoded = encode(obs)
            trace.append(encoded, idx)
            if return_raw:
                trace.raw_observations.append(obs)
            result = rpc.call("engine.applyAction", {"runId": run_id, "action": legal[idx]})
            obs = result["observation"]
            steps += 1
        outcome_raw = obs.get("outcome")
        outcome = outcome_raw if isinstance(outcome_raw, str) else "unknown"
        floor_reached = int(obs.get("floor") or 0)
    finally:
        with contextlib.suppress(Exception):
            rpc.call("engine.dispose", {"runId": run_id})

    encoded_obs, action_indices = trace.stack()
    record: GameRecord = {
        "encoded_obs": encoded_obs,
        "action_indices": action_indices,
        "n_steps": int(action_indices.shape[0]),
        "seed": int(seed),
        "character_id": character_id,
        "difficulty": difficulty,
        "outcome": outcome,
        "floor_reached": floor_reached,
    }
    if return_raw:
        record["raw_observations"] = trace.raw_observations
    return record


def write_game_npz(record: GameRecord, path: Path) -> None:
    """Serialize a ``GameRecord`` to a compressed NPZ file.

    The encoded-observation dict is flattened into ``obs_<key>`` arrays at the
    npz top level because ``np.savez`` cannot nest dicts. Metadata scalars are
    stored as 0-d arrays.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for key, arr in record["encoded_obs"].items():
        payload[f"obs_{key}"] = np.asarray(arr)
    payload["action_indices"] = np.asarray(record["action_indices"], dtype=np.int64)
    payload["n_steps"] = np.asarray(record["n_steps"], dtype=np.int64)
    payload["seed"] = np.asarray(record["seed"], dtype=np.int64)
    payload["character_id"] = np.asarray(record["character_id"])
    payload["difficulty"] = np.asarray(record["difficulty"])
    payload["outcome"] = np.asarray(record["outcome"])
    payload["floor_reached"] = np.asarray(record["floor_reached"], dtype=np.int64)
    np.savez_compressed(path, **payload)  # type: ignore[arg-type]


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
