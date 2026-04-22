"""Unit tests for :mod:`baca.bc.trainer`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces
from sb3_contrib import MaskablePPO

from baca.bc.dataset import BcDataset
from baca.bc.generate_cli import MANIFEST_NAME, SNAPSHOT_NAME
from baca.bc.generator import GameRecord, sha256_of_file, write_game_npz
from baca.bc.trainer import (
    _make_dummy_vec_env,
    build_bc_policy,
    freeze_value_head,
    save_bc_checkpoint,
    train_bc,
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
    observation_space,
)


def _realistic_encoded(n_steps: int, *, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Shape-conforming synthetic encoded observations.

    Masks are set so a couple of action slots are legal (at least 2 legal per
    row) and hand/shop/reward masks have a random-but-nonzero positive count.
    """
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


def _write_synthetic_dataset(root: Path, n_games: int, steps_per_game: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / SNAPSHOT_NAME).write_text(json.dumps(list(CARD_IDS)), encoding="utf-8")
    rng = np.random.default_rng(123)
    manifest: list[dict[str, object]] = []
    for seed in range(1, n_games + 1):
        encoded = _realistic_encoded(steps_per_game, rng=rng)
        # Make BC learnable: pick first legal action deterministically from the
        # action mask so a trained policy can actually minimize loss.
        action_mask = encoded["action_mask"]
        action_idx = np.array([int(np.argmax(row)) for row in action_mask], dtype=np.int64)
        record: GameRecord = {
            "encoded_obs": encoded,
            "action_indices": action_idx,
            "n_steps": steps_per_game,
            "seed": seed,
            "character_id": "jedrek",
            "difficulty": "normal",
            "outcome": "enemy_win",
            "floor_reached": seed % 10,
        }
        filename = f"seed_{seed:08d}.npz"
        path = root / filename
        write_game_npz(record, path)
        manifest.append(
            {
                "seed": seed,
                "filename": filename,
                "n_steps": steps_per_game,
                "outcome": "enemy_win",
                "floor_reached": seed % 10,
                "sha256": sha256_of_file(path),
            }
        )
    (root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _obs_sample() -> dict[str, np.ndarray]:
    obs: dict[str, np.ndarray] = {}
    for key, sub in observation_space().spaces.items():
        shape = (1, *(int(x) for x in (sub.shape or ())))
        obs[key] = np.zeros(shape, dtype=sub.dtype)
    return obs


def test_shouldDecreaseTrainLossOverTwoHundredStepsWhenTrainingOnFixedDataset(
    tmp_path: Path,
) -> None:
    # given
    _write_synthetic_dataset(tmp_path, n_games=1, steps_per_game=20)
    dataset = BcDataset(tmp_path)
    policy = build_bc_policy(policy_kwargs={"card_embed_dim": 16, "features_dim": 32})
    freeze_value_head(policy)

    # when - 20 epochs x 10 batches of batch_size 2 = 200 grad steps, plenty to fit 20 samples.
    metrics = train_bc(
        policy,
        dataset,
        n_epochs=20,
        batch_size=2,
        lr=3e-3,
        log_every=0,
        holdout_frac=0.0,
        seed=42,
    )

    # then
    assert metrics[0].train_loss > metrics[-1].train_loss
    assert metrics[-1].train_loss < 0.5 * metrics[0].train_loss


def test_shouldProduceValidMaskedActionWhenForwardedOnHeldOutObs(tmp_path: Path) -> None:
    # given
    _write_synthetic_dataset(tmp_path, n_games=2, steps_per_game=10)
    dataset = BcDataset(tmp_path)
    policy = build_bc_policy(policy_kwargs={"card_embed_dim": 16, "features_dim": 32})
    freeze_value_head(policy)
    train_bc(
        policy,
        dataset,
        n_epochs=2,
        batch_size=4,
        lr=3e-3,
        log_every=0,
        holdout_frac=0.0,
        seed=42,
    )

    # when
    obs = _obs_sample()
    mask = np.zeros((1, MAX_ACTIONS), dtype=bool)
    mask[0, [3, 7, 11]] = True
    action, _ = policy.predict(obs, action_masks=mask, deterministic=True)

    # then
    idx = int(action[0])
    assert mask[0, idx], f"predicted action {idx} is not in the legal mask"


def test_shouldFreezeValueNetParametersWhenFreezeCalled() -> None:
    # given
    policy = build_bc_policy(policy_kwargs={"card_embed_dim": 16, "features_dim": 32})

    # when
    frozen = freeze_value_head(policy)

    # then
    assert frozen > 0
    assert all(not p.requires_grad for p in policy.value_net.parameters())


def test_shouldRoundTripCheckpointWhenSavedAndLoadedViaMaskablePPO(tmp_path: Path) -> None:
    # given
    _write_synthetic_dataset(tmp_path, n_games=1, steps_per_game=8)
    dataset = BcDataset(tmp_path)
    policy_kwargs = {"card_embed_dim": 16, "features_dim": 32}
    policy = build_bc_policy(policy_kwargs=dict(policy_kwargs))
    freeze_value_head(policy)
    train_bc(
        policy,
        dataset,
        n_epochs=1,
        batch_size=4,
        lr=1e-3,
        log_every=0,
        holdout_frac=0.0,
        seed=42,
    )

    # when
    out_path = tmp_path / "bc.zip"
    save_bc_checkpoint(
        policy,
        observation_space(),
        spaces.Discrete(MAX_ACTIONS),
        out_path,
        policy_kwargs=policy_kwargs,
    )

    # then — external load + predict mirrors Step 8's consumption path.
    env = _make_dummy_vec_env(observation_space(), spaces.Discrete(MAX_ACTIONS))
    try:
        reloaded = MaskablePPO.load(str(out_path), env=env)
        obs = _obs_sample()
        mask = np.zeros((1, MAX_ACTIONS), dtype=bool)
        mask[0, [1, 2, 5]] = True
        action, _ = reloaded.predict(obs, action_masks=mask, deterministic=True)
        assert mask[0, int(action[0])], "reloaded policy produced out-of-mask action"
    finally:
        env.close()


def test_shouldRaiseWhenFreezeFindsNoValueParameters() -> None:
    # given
    policy = build_bc_policy(policy_kwargs={"card_embed_dim": 16, "features_dim": 32})
    policy.value_net = torch.nn.Identity()  # type: ignore[assignment]
    if hasattr(policy.mlp_extractor, "value_net"):
        policy.mlp_extractor.value_net = torch.nn.Identity()  # type: ignore[assignment]

    # when / then
    with pytest.raises(RuntimeError, match="no value-head parameters"):
        freeze_value_head(policy)
