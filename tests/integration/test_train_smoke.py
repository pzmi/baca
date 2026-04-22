"""End-to-end short training smoke + eval-callback re-verify.

Regression-catches the iter-1 ``MaskableEvalCallback`` hang by running a short
1_000-step training loop with ``n_envs=2`` and ``eval_freq`` small enough to
fire at least once. Hard timeout prevents the suite from hanging if the
callback regresses.
"""

from __future__ import annotations

import os
from pathlib import Path

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
