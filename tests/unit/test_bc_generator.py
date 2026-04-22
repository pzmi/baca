"""Unit tests for :mod:`baca.bc.generator`."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from baca.bc.generator import (
    GameRecord,
    action_to_index,
    collect_one_game,
    write_game_npz,
)
from baca.bc.heuristic_bridge import HeuristicBridge
from baca.encoder import (
    CARD_TYPES,
    MAX_ACTIONS,
    MAX_HAND_SIZE,
    MAX_REWARD_CARDS,
    MAX_SHOP_CARDS,
    PHASES,
    PLAYER_STATUSES,
    encode,
)
from baca.rpc_client import RpcClient


def _engine_dir() -> Path:
    override = os.environ.get("BACA_ENGINE_DIR")
    if override:
        return Path(override).resolve()
    return (Path(__file__).resolve().parent.parent.parent.parent / "slay-the-ceper").resolve()


def _engine_present() -> bool:
    return (_engine_dir() / "scripts" / "rpc-server.js").is_file()


def _node_present() -> bool:
    return shutil.which("node") is not None


def test_shouldMapActionToLegalIndexWhenActionEqualsFirstLegal() -> None:
    # given
    legal: list[dict[str, Any]] = [
        {"type": "end_turn"},
        {"type": "play_card", "handIndex": 0},
    ]

    # when
    idx = action_to_index({"type": "end_turn"}, legal)

    # then
    assert idx == 0


def test_shouldMapActionToSecondLegalWhenActionEqualsSecond() -> None:
    # given
    legal: list[dict[str, Any]] = [
        {"type": "end_turn"},
        {"type": "play_card", "handIndex": 2},
    ]

    # when
    idx = action_to_index({"type": "play_card", "handIndex": 2}, legal)

    # then
    assert idx == 1


def test_shouldRaiseWhenHeuristicActionNotInLegalList() -> None:
    # given
    legal: list[dict[str, Any]] = [{"type": "end_turn"}]

    # when / then
    with pytest.raises(ValueError, match="not in legal list"):
        action_to_index({"type": "play_card", "handIndex": 0}, legal)


def test_shouldMapStructurallyEqualDictWhenKeysInDifferentOrder() -> None:
    # given — HeuristicBot returns fields in whatever order; canonical JSON
    #         ordering must not change the match.
    legal: list[dict[str, Any]] = [{"type": "play_card", "handIndex": 3}]
    action_with_reordered_keys: dict[str, Any] = {"handIndex": 3, "type": "play_card"}

    # when
    idx = action_to_index(action_with_reordered_keys, legal)

    # then
    assert idx == 0


def _fake_encoded_obs(n_steps: int) -> dict[str, np.ndarray]:
    """Synthetic encoded-observation stack with the right dtypes/shapes."""
    return {
        "player": np.zeros((n_steps, 4 + len(PLAYER_STATUSES)), dtype=np.float32),
        "enemy": np.zeros((n_steps, 5), dtype=np.float32),
        "enemy_present": np.zeros((n_steps, 1), dtype=np.float32),
        "phase": np.zeros((n_steps, len(PHASES)), dtype=np.float32),
        "run": np.zeros((n_steps, 4), dtype=np.float32),
        "hand": np.zeros((n_steps, MAX_HAND_SIZE, 2 + len(CARD_TYPES)), dtype=np.float32),
        "hand_mask": np.zeros((n_steps, MAX_HAND_SIZE), dtype=np.int8),
        "hand_ids": np.arange(n_steps * MAX_HAND_SIZE, dtype=np.int64).reshape(
            n_steps, MAX_HAND_SIZE
        ),
        "reward_ids": np.zeros((n_steps, MAX_REWARD_CARDS), dtype=np.int64),
        "reward_mask": np.zeros((n_steps, MAX_REWARD_CARDS), dtype=np.int8),
        "reward_costs": np.zeros((n_steps, MAX_REWARD_CARDS), dtype=np.float32),
        "shop_ids": np.zeros((n_steps, MAX_SHOP_CARDS), dtype=np.int64),
        "shop_mask": np.zeros((n_steps, MAX_SHOP_CARDS), dtype=np.int8),
        "shop_costs": np.zeros((n_steps, MAX_SHOP_CARDS), dtype=np.float32),
        "action_mask": np.zeros((n_steps, MAX_ACTIONS), dtype=np.int8),
    }


def test_shouldWriteNpzRoundTrippingEncodedObsAndActionsWhenWritten(tmp_path: Path) -> None:
    # given
    n_steps = 7
    encoded_obs = _fake_encoded_obs(n_steps)
    action_indices = np.arange(n_steps, dtype=np.int64)
    record: GameRecord = {
        "encoded_obs": encoded_obs,
        "action_indices": action_indices,
        "n_steps": n_steps,
        "seed": 42,
        "character_id": "jedrek",
        "difficulty": "normal",
        "outcome": "enemy_win",
        "floor_reached": 3,
    }
    target = tmp_path / "seed_00000042.npz"

    # when
    write_game_npz(record, target)
    with np.load(target) as data:
        loaded_keys = set(data.files)
        reloaded_obs = {
            name[len("obs_") :]: np.asarray(data[name])
            for name in data.files
            if name.startswith("obs_")
        }
        reloaded_actions = np.asarray(data["action_indices"])
        reloaded_seed = int(np.asarray(data["seed"]))
        reloaded_n_steps = int(np.asarray(data["n_steps"]))
        reloaded_outcome = str(np.asarray(data["outcome"]))
        reloaded_floor = int(np.asarray(data["floor_reached"]))

    # then
    assert target.is_file()
    for key, arr in encoded_obs.items():
        assert f"obs_{key}" in loaded_keys
        assert reloaded_obs[key].shape == arr.shape
        np.testing.assert_array_equal(reloaded_obs[key], arr)
    np.testing.assert_array_equal(reloaded_actions, action_indices)
    assert reloaded_seed == 42
    assert reloaded_n_steps == n_steps
    assert reloaded_outcome == "enemy_win"
    assert reloaded_floor == 3


@pytest.mark.skipif(not _node_present(), reason="node binary not found")
@pytest.mark.skipif(not _engine_present(), reason="engine checkout missing")
@pytest.mark.timeout(60)
def test_shouldMatchEncoderOutputWhenRoundTrippingGameRecord() -> None:
    # given
    engine_dir = _engine_dir()
    rpc = RpcClient(engine_dir=engine_dir)
    bridge = HeuristicBridge(engine_dir=engine_dir)
    try:
        # when
        record = collect_one_game(rpc, bridge, seed=1, return_raw=True)
    finally:
        bridge.close()
        rpc.close()

    # then
    raw = record.get("raw_observations") or []
    assert record["n_steps"] == len(raw) == record["action_indices"].shape[0]
    assert record["n_steps"] > 0, "game must have at least one decision step"
    n_checks = min(3, record["n_steps"])
    for step in range(n_checks):
        fresh_encoded = encode(raw[step])
        for key, expected in fresh_encoded.items():
            stored = record["encoded_obs"][key][step]
            np.testing.assert_array_equal(stored, expected)


@pytest.mark.skipif(not _node_present(), reason="node binary not found")
@pytest.mark.skipif(not _engine_present(), reason="engine checkout missing")
@pytest.mark.timeout(60)
def test_shouldKeepActionIndicesInRangeWhenGeneratingGame() -> None:
    # given
    engine_dir = _engine_dir()
    rpc = RpcClient(engine_dir=engine_dir)
    bridge = HeuristicBridge(engine_dir=engine_dir)

    # when
    try:
        record = collect_one_game(rpc, bridge, seed=2)
    finally:
        bridge.close()
        rpc.close()

    # then
    actions = record["action_indices"]
    assert actions.dtype == np.int64
    assert actions.shape == (record["n_steps"],)
    assert int(actions.min(initial=0)) >= 0
    assert int(actions.max(initial=-1)) < MAX_ACTIONS
