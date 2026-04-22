"""Vec-env integration smoke: prove SubprocVecEnv runs end-to-end against engine.

Addresses the v0 gap where vec-env wiring had unit coverage only. Points
:class:`SubprocVecEnv` at ``n_envs=2`` real RPC-backed workers, steps for a few
dozen actions picking the first legal action per worker, and checks that the
batched observation dict stacks correctly.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from baca import train as train_module
from baca.train import TrainConfig


def _engine_present() -> bool:
    override = os.environ.get("BACA_ENGINE_DIR")
    base = Path(override) if override else Path(__file__).resolve().parents[3] / "slay-the-ceper"
    return (base / "scripts" / "rpc-server.js").is_file()


def _engine_dir() -> Path | None:
    override = os.environ.get("BACA_ENGINE_DIR")
    return Path(override).resolve() if override else None


@pytest.mark.skipif(not _engine_present(), reason="engine checkout missing")
def test_shouldStepAllEnvsWhenSubprocVecEnvConstructedWithNEnvsTwo() -> None:
    # given
    cfg = TrainConfig(
        total_timesteps=0,
        character_id="jedrek",
        difficulty="normal",
        base_seed=1,
        checkpoint_dir=Path("checkpoints_test"),
        engine_dir=_engine_dir(),
        learning_rate=3e-4,
        n_steps=16,
        batch_size=16,
        n_envs=2,
        seed=1,
        reward_shape="none",
    )
    vec = train_module._build_train_vec_env(cfg)
    try:
        # when
        obs = vec.reset()
        for _ in range(30):
            masks = vec.env_method("action_masks")
            actions = np.array([int(np.argmax(m)) for m in masks], dtype=np.int64)
            obs, _rewards, _dones, _infos = vec.step(actions)

        # then
        assert isinstance(obs, dict)
        assert obs["player"].shape[0] == 2
        assert obs["hand_ids"].shape == (2, 10)
        assert obs["reward_ids"].shape == (2, 3)
        assert obs["shop_ids"].shape == (2, 3)
        assert obs["action_mask"].shape == (2, 32)
    finally:
        vec.close()
