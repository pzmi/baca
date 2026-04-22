"""Unit tests for :mod:`baca.bc.value_pretrain`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from baca.bc.dataset import BcDataset
from baca.bc.generate_cli import MANIFEST_NAME, SNAPSHOT_NAME
from baca.bc.generator import GameRecord, sha256_of_file, write_game_npz
from baca.bc.trainer import build_bc_policy, freeze_value_head
from baca.bc.value_pretrain import (
    build_rewards_from_manifest,
    compute_returns,
    freeze_policy_side,
    train_value_head,
)
from baca.encoder import (
    CARD_IDS,
    CARD_TYPES,
    MAX_ACTIONS,
    MAX_HAND_SIZE,
    MAX_REWARD_CARDS,
    MAX_SHOP_CARDS,
    PHASES,
    PLAYER_STATUSES,
)


def _realistic_encoded(n_steps: int, *, rng: np.random.Generator) -> dict[str, np.ndarray]:
    legal_count = rng.integers(low=2, high=8, size=n_steps)
    action_mask = np.zeros((n_steps, MAX_ACTIONS), dtype=np.int8)
    for i, k in enumerate(legal_count):
        action_mask[i, : int(k)] = 1
    return {
        "player": rng.random((n_steps, 4 + len(PLAYER_STATUSES)), dtype=np.float32),
        "enemy": rng.random((n_steps, 5), dtype=np.float32),
        "enemy_present": rng.random((n_steps, 1), dtype=np.float32),
        "phase": rng.random((n_steps, len(PHASES)), dtype=np.float32),
        "run": rng.random((n_steps, 4), dtype=np.float32),
        "hand": rng.random((n_steps, MAX_HAND_SIZE, 2 + len(CARD_TYPES)), dtype=np.float32),
        "hand_mask": rng.integers(low=0, high=2, size=(n_steps, MAX_HAND_SIZE), dtype=np.int8),
        "hand_ids": rng.integers(
            low=0, high=len(CARD_IDS), size=(n_steps, MAX_HAND_SIZE), dtype=np.int64
        ),
        "reward_ids": rng.integers(
            low=0, high=len(CARD_IDS), size=(n_steps, MAX_REWARD_CARDS), dtype=np.int64
        ),
        "reward_mask": rng.integers(low=0, high=2, size=(n_steps, MAX_REWARD_CARDS), dtype=np.int8),
        "reward_costs": rng.random((n_steps, MAX_REWARD_CARDS), dtype=np.float32),
        "shop_ids": rng.integers(
            low=0, high=len(CARD_IDS), size=(n_steps, MAX_SHOP_CARDS), dtype=np.int64
        ),
        "shop_mask": rng.integers(low=0, high=2, size=(n_steps, MAX_SHOP_CARDS), dtype=np.int8),
        "shop_costs": rng.random((n_steps, MAX_SHOP_CARDS), dtype=np.float32),
        "action_mask": action_mask,
    }


def _write_value_dataset(root: Path, games: list[tuple[int, int, str, int]]) -> None:
    """Write a synthetic dataset. ``games`` is ``(seed, n_steps, outcome, floor)``."""
    root.mkdir(parents=True, exist_ok=True)
    (root / SNAPSHOT_NAME).write_text(json.dumps(list(CARD_IDS)), encoding="utf-8")
    rng = np.random.default_rng(321)
    manifest: list[dict[str, object]] = []
    for seed, n_steps, outcome, floor in games:
        encoded = _realistic_encoded(n_steps, rng=rng)
        action_idx = np.array(
            [int(np.argmax(row)) for row in encoded["action_mask"]], dtype=np.int64
        )
        record: GameRecord = {
            "encoded_obs": encoded,
            "action_indices": action_idx,
            "n_steps": n_steps,
            "seed": seed,
            "character_id": "jedrek",
            "difficulty": "normal",
            "outcome": outcome,
            "floor_reached": floor,
        }
        filename = f"seed_{seed:08d}.npz"
        path = root / filename
        write_game_npz(record, path)
        manifest.append(
            {
                "seed": seed,
                "filename": filename,
                "n_steps": n_steps,
                "outcome": outcome,
                "floor_reached": floor,
                "sha256": sha256_of_file(path),
            }
        )
    (root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_shouldComputeTrajectoryReturnsWhenGivenTerminalRewardOnly() -> None:
    # given
    rewards = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    # when
    returns = compute_returns(rewards, gamma=0.99)

    # then
    expected = np.array([0.99**3, 0.99**2, 0.99, 1.0], dtype=np.float32)
    np.testing.assert_allclose(returns, expected, atol=1e-6)

    # empty case
    empty = compute_returns(np.array([], dtype=np.float32), gamma=0.99)
    assert empty.shape == (0,)
    assert empty.dtype == np.float32


def test_shouldComputeTerminalRewardMatchingEnvShapingWhenWinAtFloor() -> None:
    # given / when — winning at floor 15 gives the full shaping reward.
    rewards_win = build_rewards_from_manifest(n_steps=5, outcome="player_win", floor_reached=15)

    # then
    assert rewards_win.shape == (5,)
    assert rewards_win.dtype == np.float32
    assert np.all(rewards_win[:-1] == 0.0)
    assert rewards_win[-1] == np.float32(0.1 * 1.0 + 0.9 * 1.0)

    # and — losing at floor 7 should give only the floor-shaping part.
    rewards_lose = build_rewards_from_manifest(n_steps=5, outcome="player_lose", floor_reached=7)
    assert np.all(rewards_lose[:-1] == 0.0)
    expected_terminal = np.float32(0.1 * (7.0 / 15.0) + 0.9 * 0.0)
    np.testing.assert_allclose(rewards_lose[-1], expected_terminal, atol=1e-6)


def test_shouldProduceZeroTerminalRewardWhenTruncated() -> None:
    # given / when
    rewards = build_rewards_from_manifest(n_steps=5, outcome="truncated", floor_reached=10)

    # then — truncated episodes contribute nothing to value targets.
    assert rewards.shape == (5,)
    assert np.all(rewards == 0.0)

    # and — unknown / missing outcomes are treated identically.
    rewards_unknown = build_rewards_from_manifest(n_steps=3, outcome="unknown", floor_reached=4)
    assert np.all(rewards_unknown == 0.0)


def test_shouldFreezeAllButValueHeadWhenFreezePolicySideCalled() -> None:
    # given
    policy = build_bc_policy(policy_kwargs={"card_embed_dim": 16, "features_dim": 32})

    # when
    frozen = freeze_policy_side(policy)

    # then — features extractor, mlp policy branch, action head all frozen.
    assert all(not p.requires_grad for p in policy.features_extractor.parameters())
    assert all(not p.requires_grad for p in policy.action_net.parameters())
    policy_branch = policy.mlp_extractor.policy_net
    assert all(not p.requires_grad for p in policy_branch.parameters())

    # and — both value-side modules stay trainable.
    assert all(p.requires_grad for p in policy.value_net.parameters())
    assert all(p.requires_grad for p in policy.mlp_extractor.value_net.parameters())

    trainable_count = sum(1 for p in policy.parameters() if p.requires_grad)
    value_side_count = sum(1 for _ in policy.value_net.parameters()) + sum(
        1 for _ in policy.mlp_extractor.value_net.parameters()
    )
    assert trainable_count == value_side_count
    assert frozen == sum(1 for p in policy.parameters() if not p.requires_grad)

    # and — the count matches the INVERSE of freeze_value_head on a fresh policy.
    twin = build_bc_policy(policy_kwargs={"card_embed_dim": 16, "features_dim": 32})
    value_frozen = freeze_value_head(twin)
    assert trainable_count == value_frozen


def test_shouldDecreaseValueLossAcrossEpochsWhenTrainingOnFixedDataset(
    tmp_path: Path,
) -> None:
    # given — 3 games with varied outcomes so returns are not all zero.
    _write_value_dataset(
        tmp_path,
        [
            (1, 6, "player_win", 15),
            (2, 6, "player_lose", 7),
            (3, 6, "player_win", 12),
        ],
    )
    dataset = BcDataset(tmp_path)
    policy = build_bc_policy(policy_kwargs={"card_embed_dim": 16, "features_dim": 32})
    freeze_policy_side(policy)

    # when
    metrics = train_value_head(
        policy,
        dataset,
        n_epochs=5,
        batch_size=4,
        lr=1e-2,
        gamma=0.99,
        log_every=0,
        seed=42,
    )

    # then
    assert len(metrics) == 5
    assert metrics[-1].train_loss < 0.5 * metrics[0].train_loss


def test_shouldNotModifyPolicyHeadWeightsWhenValuePretrainRuns(tmp_path: Path) -> None:
    # given
    _write_value_dataset(
        tmp_path,
        [
            (1, 5, "player_win", 15),
            (2, 5, "player_lose", 3),
        ],
    )
    dataset = BcDataset(tmp_path)
    policy = build_bc_policy(policy_kwargs={"card_embed_dim": 16, "features_dim": 32})
    freeze_policy_side(policy)

    embedding_snapshot = policy.features_extractor._embedding.weight.detach().clone()  # type: ignore[operator,union-attr]
    action_snapshot = policy.action_net.weight.detach().clone()  # type: ignore[operator]
    policy_branch_snapshot = {
        name: param.detach().clone()
        for name, param in policy.mlp_extractor.policy_net.named_parameters()
    }

    # when
    train_value_head(
        policy,
        dataset,
        n_epochs=2,
        batch_size=4,
        lr=1e-2,
        gamma=0.99,
        log_every=0,
        seed=42,
    )

    # then — bit-exact equality on frozen weights (R4 contract).
    assert torch.equal(
        policy.features_extractor._embedding.weight.detach().cpu(),  # type: ignore[operator,union-attr]
        embedding_snapshot,
    )
    assert torch.equal(policy.action_net.weight.detach().cpu(), action_snapshot)  # type: ignore[operator]
    for name, param in policy.mlp_extractor.policy_net.named_parameters():
        assert torch.equal(param.detach().cpu(), policy_branch_snapshot[name]), (
            f"policy_net param {name} changed under value pretrain"
        )
