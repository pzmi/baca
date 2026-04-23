"""End-to-end short training smoke + eval-callback re-verify.

Regression-catches the iter-1 ``MaskableEvalCallback`` hang by running a short
1_000-step training loop with ``n_envs=2`` and ``eval_freq`` small enough to
fire at least once. Hard timeout prevents the suite from hanging if the
callback regresses.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from gymnasium import spaces

from baca import train as train_module
from baca.bc.trainer import build_bc_policy, save_bc_checkpoint
from baca.encoder import MAX_ACTIONS, observation_space
from baca.train import TrainConfig


def _engine_present() -> bool:
    override = os.environ.get("BACA_ENGINE_DIR")
    base = Path(override) if override else Path(__file__).resolve().parents[3] / "slay-the-ceper"
    return (base / "scripts" / "rpc-server.js").is_file()


def _engine_dir() -> Path | None:
    override = os.environ.get("BACA_ENGINE_DIR")
    return Path(override).resolve() if override else None


def _node_present() -> bool:
    return shutil.which("node") is not None


@pytest.mark.skipif(not _engine_present(), reason="engine checkout missing")
@pytest.mark.timeout(300)
def test_shouldFireEvalCallbackOnceWhenTraining1000StepsWithNEnvsTwo(tmp_path: Path) -> None:
    # given
    cfg = TrainConfig(
        total_timesteps=1_000,
        character_id="jedrek",
        difficulty="normal",
        base_seed=1,
        checkpoint_dir=tmp_path / "run",
        engine_dir=_engine_dir(),
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_envs=2,
        seed=42,
        reward_shape="floor",
        eval_freq=500,
        eval_episodes=2,
        checkpoint_freq=1_000_000,
    )

    # when
    out = train_module.train(cfg)

    # then
    assert out.exists()
    assert (cfg.checkpoint_dir / "best").exists()


@pytest.mark.skipif(not _engine_present() or not _node_present(), reason="engine/node missing")
@pytest.mark.timeout(300)
def test_shouldLoadBcCheckpointWhenBcInitFlagSet(tmp_path: Path) -> None:
    # given — minimal BC checkpoint from an untrained MaskableBacaPolicy
    bc_dir = tmp_path / "bc"
    bc_path = bc_dir / "bc_model.zip"
    obs_space = observation_space()
    act_space: spaces.Space[int] = spaces.Discrete(MAX_ACTIONS)
    policy = build_bc_policy(obs_space, act_space)
    save_bc_checkpoint(policy, obs_space, act_space, bc_path)

    cfg = TrainConfig(
        total_timesteps=500,
        character_id="jedrek",
        difficulty="normal",
        base_seed=1,
        checkpoint_dir=tmp_path / "run",
        engine_dir=_engine_dir(),
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_envs=1,
        seed=42,
        reward_shape="floor",
        eval_freq=1_000_000,
        eval_episodes=1,
        checkpoint_freq=1_000_000,
        bc_init=bc_path,
        initial_lr_scale=0.33,
        warmup_frac=0.1,
        clip_range=0.1,
    )

    # when
    out = train_module.train(cfg)

    # then
    assert out.exists()
    assert out == cfg.checkpoint_dir / "baca-latest.zip"
