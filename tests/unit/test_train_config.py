"""Unit tests for :mod:`baca.train` config + LR schedule factory."""

from __future__ import annotations

from pathlib import Path

from baca.train import TrainConfig, _linear_schedule, _parse_args


def test_shouldReturnStartRateWhenProgressRemainingIsOne() -> None:
    # given
    schedule = _linear_schedule(3e-4, 1e-4)

    # when
    rate = schedule(1.0)

    # then
    assert rate == 3e-4


def test_shouldReturnFinalRateWhenProgressRemainingIsZero() -> None:
    # given
    schedule = _linear_schedule(3e-4, 1e-4)

    # when
    rate = schedule(0.0)

    # then
    assert rate == 1e-4


def test_shouldInterpolateLinearlyWhenProgressRemainingIsHalf() -> None:
    # given
    schedule = _linear_schedule(3e-4, 1e-4)

    # when
    rate = schedule(0.5)

    # then
    assert rate == (3e-4 + 1e-4) / 2


def test_shouldExposeSparseRewardDefaultsWhenConfigConstructedWithoutOverrides() -> None:
    # given
    cfg = TrainConfig(
        total_timesteps=1,
        character_id="jedrek",
        difficulty="normal",
        base_seed=None,
        checkpoint_dir=Path("checkpoints"),
        engine_dir=None,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
    )

    # when / then
    assert cfg.ent_coef == 0.01
    assert cfg.n_epochs == 10
    assert cfg.gamma == 0.99
    assert cfg.gae_lambda == 0.95
    assert cfg.clip_range == 0.2
    assert cfg.lr_final == 1e-4
    assert cfg.eval_freq == 10_000
    assert cfg.eval_episodes == 50
    assert cfg.checkpoint_freq == 20_000
    assert cfg.seed == 42
    assert cfg.n_envs == 8
    assert cfg.reward_shape == "floor"
    assert cfg.card_embed_dim == 32
    assert cfg.features_dim == 64


def test_shouldParseTrainingArgsWhenAllFlagsProvided() -> None:
    # given
    argv = [
        "--total-timesteps",
        "500",
        "--character-id",
        "jedrek",
        "--difficulty",
        "hard",
        "--base-seed",
        "7",
        "--checkpoint-dir",
        "out",
        "--learning-rate",
        "2e-4",
        "--n-steps",
        "128",
        "--batch-size",
        "32",
        "--ent-coef",
        "0.02",
        "--n-epochs",
        "5",
        "--gamma",
        "0.99",
        "--gae-lambda",
        "0.9",
        "--clip-range",
        "0.1",
        "--lr-final",
        "5e-5",
        "--eval-freq",
        "1000",
        "--eval-episodes",
        "11",
        "--checkpoint-freq",
        "2500",
        "--seed",
        "123",
        "--n-envs",
        "4",
    ]

    # when
    cfg = _parse_args(argv)

    # then
    assert cfg.total_timesteps == 500
    assert cfg.character_id == "jedrek"
    assert cfg.difficulty == "hard"
    assert cfg.base_seed == 7
    assert cfg.checkpoint_dir == Path("out")
    assert cfg.engine_dir is None
    assert cfg.learning_rate == 2e-4
    assert cfg.n_steps == 128
    assert cfg.batch_size == 32
    assert cfg.ent_coef == 0.02
    assert cfg.n_epochs == 5
    assert cfg.gamma == 0.99
    assert cfg.gae_lambda == 0.9
    assert cfg.clip_range == 0.1
    assert cfg.lr_final == 5e-5
    assert cfg.eval_freq == 1000
    assert cfg.eval_episodes == 11
    assert cfg.checkpoint_freq == 2500
    assert cfg.seed == 123
    assert cfg.n_envs == 4


def test_shouldDefaultEngineDirToNoneWhenFlagOmitted() -> None:
    # given
    argv: list[str] = []

    # when
    cfg = _parse_args(argv)

    # then
    assert cfg.engine_dir is None
    assert cfg.base_seed is None
    assert cfg.learning_rate == 3e-4
    assert cfg.lr_final == 1e-4


def test_shouldDefaultGammaToNineNineWhenConfigConstructedWithoutOverrides() -> None:
    # given
    cfg = TrainConfig(
        total_timesteps=1,
        character_id="jedrek",
        difficulty="normal",
        base_seed=None,
        checkpoint_dir=Path("checkpoints"),
        engine_dir=None,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
    )

    # when / then
    assert cfg.gamma == 0.99


def test_shouldDefaultRewardShapeToFloorWhenConfigConstructedWithoutOverrides() -> None:
    # given
    cfg = TrainConfig(
        total_timesteps=1,
        character_id="jedrek",
        difficulty="normal",
        base_seed=None,
        checkpoint_dir=Path("checkpoints"),
        engine_dir=None,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
    )

    # when / then
    assert cfg.reward_shape == "floor"


def test_shouldParseRewardShapeChoiceWhenFlagProvided() -> None:
    # given
    argv = ["--reward-shape", "none"]

    # when
    cfg = _parse_args(argv)

    # then
    assert cfg.reward_shape == "none"


def test_shouldParseCardEmbedDimWhenFlagProvided() -> None:
    # given
    argv = ["--card-embed-dim", "48", "--features-dim", "96"]

    # when
    cfg = _parse_args(argv)

    # then
    assert cfg.card_embed_dim == 48
    assert cfg.features_dim == 96
