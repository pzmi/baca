"""Unit tests for :mod:`baca.bc.dataset`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

from baca.bc.dataset import BcDataset
from baca.bc.generate_cli import MANIFEST_NAME, SNAPSHOT_NAME
from baca.bc.generator import GameRecord, sha256_of_file, write_game_npz
from baca.encoder import (
    CARD_IDS,
    CARD_TYPES,
    MAX_ACTIONS,
    MAX_HAND_SIZE,
    MAX_REWARD_CARDS,
    MAX_SHOP_CARDS,
    PHASES,
    PLAYER_STATUSES,
    observation_space,
)


def _fake_encoded_obs(n_steps: int, *, fill: int) -> dict[str, np.ndarray]:
    return {
        "player": np.full((n_steps, 4 + len(PLAYER_STATUSES)), fill, dtype=np.float32),
        "enemy": np.full((n_steps, 5), fill, dtype=np.float32),
        "enemy_present": np.full((n_steps, 1), fill, dtype=np.float32),
        "phase": np.full((n_steps, len(PHASES)), fill, dtype=np.float32),
        "run": np.full((n_steps, 4), fill, dtype=np.float32),
        "hand": np.full((n_steps, MAX_HAND_SIZE, 2 + len(CARD_TYPES)), fill, dtype=np.float32),
        "hand_mask": np.full((n_steps, MAX_HAND_SIZE), fill % 2, dtype=np.int8),
        "hand_ids": np.full((n_steps, MAX_HAND_SIZE), fill, dtype=np.int64),
        "reward_ids": np.full((n_steps, MAX_REWARD_CARDS), fill, dtype=np.int64),
        "reward_mask": np.full((n_steps, MAX_REWARD_CARDS), fill % 2, dtype=np.int8),
        "reward_costs": np.full((n_steps, MAX_REWARD_CARDS), fill, dtype=np.float32),
        "shop_ids": np.full((n_steps, MAX_SHOP_CARDS), fill, dtype=np.int64),
        "shop_mask": np.full((n_steps, MAX_SHOP_CARDS), fill % 2, dtype=np.int8),
        "shop_costs": np.full((n_steps, MAX_SHOP_CARDS), fill, dtype=np.float32),
        "action_mask": np.full((n_steps, MAX_ACTIONS), fill % 2, dtype=np.int8),
    }


def _make_dataset(root: Path, game_specs: list[tuple[int, int]]) -> None:
    """Write fake NPZ games + manifest + snapshot into ``root``.

    ``game_specs`` is a list of ``(seed, n_steps)``.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / SNAPSHOT_NAME).write_text(json.dumps(list(CARD_IDS)), encoding="utf-8")
    manifest: list[dict[str, object]] = []
    for seed, n_steps in game_specs:
        filename = f"seed_{seed:08d}.npz"
        path = root / filename
        record: GameRecord = {
            "encoded_obs": _fake_encoded_obs(n_steps, fill=seed),
            "action_indices": np.arange(n_steps, dtype=np.int64) % MAX_ACTIONS,
            "n_steps": n_steps,
            "seed": seed,
            "character_id": "jedrek",
            "difficulty": "normal",
            "outcome": "enemy_win",
            "floor_reached": seed % 15,
        }
        write_game_npz(record, path)
        manifest.append(
            {
                "seed": seed,
                "filename": filename,
                "n_steps": n_steps,
                "outcome": "enemy_win",
                "floor_reached": seed % 15,
                "sha256": sha256_of_file(path),
            }
        )
    (root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_shouldIterateAllSamplesWhenBatchSizeOne(tmp_path: Path) -> None:
    # given
    _make_dataset(tmp_path, [(1, 3), (2, 4), (3, 2)])
    dataset = BcDataset(tmp_path)

    # when
    batches = list(dataset.iter_batches(batch_size=1, shuffle=False))

    # then
    assert len(dataset) == 9
    assert len(batches) == 9
    for batch in batches:
        assert batch["action_indices"].shape == (1,)


def test_shouldShuffleBetweenEpochsWhenSeededDifferently(tmp_path: Path) -> None:
    # given
    _make_dataset(tmp_path, [(1, 4), (2, 5), (3, 6)])
    dataset = BcDataset(tmp_path)

    # when
    first_a = next(iter(dataset.iter_batches(batch_size=3, shuffle=True, seed=1)))
    first_b = next(iter(dataset.iter_batches(batch_size=3, shuffle=True, seed=999)))
    first_a_repeat = next(iter(dataset.iter_batches(batch_size=3, shuffle=True, seed=1)))

    # then — same seed reproduces exactly.
    torch.testing.assert_close(first_a["action_indices"], first_a_repeat["action_indices"])
    # Different seeds yield different orderings with high probability; if they
    # tie by coincidence on this fixed fake dataset, keep the test robust by
    # comparing the distribution across multiple seeds instead of one.
    if torch.equal(first_a["action_indices"], first_b["action_indices"]):
        first_c = next(iter(dataset.iter_batches(batch_size=3, shuffle=True, seed=7)))
        assert not torch.equal(first_a["action_indices"], first_c["action_indices"])


def test_shouldYieldTensorShapesMatchingObservationSpaceWhenIterating(tmp_path: Path) -> None:
    # given
    _make_dataset(tmp_path, [(1, 5), (2, 5)])
    dataset = BcDataset(tmp_path)
    space = observation_space()

    # when
    batch = next(iter(dataset.iter_batches(batch_size=4, shuffle=False)))

    # then
    for key, subspace in space.spaces.items():
        expected_shape = (4, *(int(x) for x in (subspace.shape or ())))
        tensor = batch[key]
        assert tensor.shape == expected_shape, (
            f"batch[{key!r}] shape {tensor.shape} != {expected_shape}"
        )
        if isinstance(subspace, spaces.MultiBinary):
            assert tensor.dtype == torch.bool
        elif key.endswith("_ids"):
            assert tensor.dtype == torch.long
        else:
            assert tensor.dtype == torch.float32
    assert batch["action_indices"].shape == (4,)
    assert batch["action_indices"].dtype == torch.long


def test_shouldRaiseWhenCardIdsSnapshotMismatches(tmp_path: Path) -> None:
    # given
    _make_dataset(tmp_path, [(1, 2)])
    (tmp_path / SNAPSHOT_NAME).write_text(json.dumps(["UNKNOWN", "bogus"]), encoding="utf-8")

    # when / then
    with pytest.raises(ValueError, match="CARD_IDS mismatch"):
        BcDataset(tmp_path)


def test_shouldRaiseWhenSnapshotFileMissing(tmp_path: Path) -> None:
    # given
    _make_dataset(tmp_path, [(1, 2)])
    (tmp_path / SNAPSHOT_NAME).unlink()

    # when / then
    with pytest.raises(ValueError, match="missing"):
        BcDataset(tmp_path)


def test_shouldRaiseWhenBatchSizeNonPositive(tmp_path: Path) -> None:
    # given
    _make_dataset(tmp_path, [(1, 3)])
    dataset = BcDataset(tmp_path)

    # when / then
    with pytest.raises(ValueError, match="batch_size"):
        list(dataset.iter_batches(batch_size=0))


def test_shouldYieldRemainderBatchWhenSamplesDontDivideEvenly(tmp_path: Path) -> None:
    # given
    _make_dataset(tmp_path, [(1, 3), (2, 4)])
    dataset = BcDataset(tmp_path)

    # when
    batches = list(dataset.iter_batches(batch_size=3, shuffle=False))

    # then — 7 samples / 3 per batch → 2 full + 1 remainder of size 1
    assert sum(int(b["action_indices"].shape[0]) for b in batches) == 7
    assert batches[-1]["action_indices"].shape[0] == 1
