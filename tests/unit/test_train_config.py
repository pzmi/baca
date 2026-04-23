"""Unit tests for :mod:`baca.train` config + LR schedule factory."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from baca.bc.trainer import _DummyEnv, build_bc_policy, save_bc_checkpoint
from baca.encoder import MAX_ACTIONS, observation_space
from baca.train import (
    TrainConfig,
    _linear_schedule,
    _load_bc_model,
    _parse_args,
    _warmup_schedule,
)


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


def test_shouldParseBcInitPathWhenFlagProvided() -> None:
    # given
    argv = ["--bc-init", "foo.zip"]

    # when
    cfg = _parse_args(argv)

    # then
    assert cfg.bc_init == Path("foo.zip")


def test_shouldDefaultBcInitToNoneWhenFlagOmitted() -> None:
    # given
    argv: list[str] = []

    # when
    cfg = _parse_args(argv)

    # then
    assert cfg.bc_init is None


def test_shouldDefaultInitialLrScaleToOneWhenNotProvided() -> None:
    # given
    argv_with_bc = ["--bc-init", "foo.zip"]
    argv_without_bc: list[str] = []

    # when
    cfg_with_bc = _parse_args(argv_with_bc)
    cfg_without_bc = _parse_args(argv_without_bc)

    # then
    assert cfg_with_bc.initial_lr_scale == 1.0
    assert cfg_without_bc.initial_lr_scale == 1.0


def test_shouldParseWarmupFracWhenFlagProvided() -> None:
    # given
    argv = ["--warmup-frac", "0.25", "--initial-lr-scale", "0.33"]

    # when
    cfg = _parse_args(argv)

    # then
    assert cfg.warmup_frac == 0.25
    assert cfg.initial_lr_scale == 0.33


def test_shouldDefaultWarmupFracToPointOneWhenFlagOmitted() -> None:
    # given
    argv: list[str] = []

    # when
    cfg = _parse_args(argv)

    # then
    assert cfg.warmup_frac == 0.1


def test_shouldDefaultResetNumTimestepsTrueWhenFlagOmitted() -> None:
    # given
    argv: list[str] = []

    # when
    cfg = _parse_args(argv)

    # then
    assert cfg.reset_num_timesteps is True


def test_shouldPassThroughResetNumTimestepsWhenFlagSet() -> None:
    # given
    argv_reset = ["--reset-num-timesteps"]
    argv_no_reset = ["--no-reset-num-timesteps"]

    # when
    cfg_reset = _parse_args(argv_reset)
    cfg_no_reset = _parse_args(argv_no_reset)

    # then
    assert cfg_reset.reset_num_timesteps is True
    assert cfg_no_reset.reset_num_timesteps is False


def test_shouldReturnBaseScheduleWhenInitialScaleIsOne() -> None:
    # given
    base = _linear_schedule(3e-4, 1e-4)
    warmup = _warmup_schedule(3e-4, 1e-4, warmup_frac=0.1, initial_scale=1.0)

    # when / then
    for p in (1.0, 0.9, 0.5, 0.1, 0.0):
        assert warmup(p) == base(p)


def test_shouldReturnBaseScheduleWhenWarmupFracIsZero() -> None:
    # given
    base = _linear_schedule(3e-4, 1e-4)
    warmup = _warmup_schedule(3e-4, 1e-4, warmup_frac=0.0, initial_scale=0.33)

    # when / then
    for p in (1.0, 0.9, 0.5, 0.1, 0.0):
        assert warmup(p) == base(p)


def test_shouldWarmupScheduleReducesLRWhenCalledEarly() -> None:
    # given
    lr_start = 3e-4
    lr_final = 1e-4
    warmup_frac = 0.1
    initial_scale = 0.33
    base = _linear_schedule(lr_start, lr_final)
    schedule = _warmup_schedule(lr_start, lr_final, warmup_frac, initial_scale)

    # when / then
    # Start of training: full warmup scaling applied to the base rate.
    assert schedule(1.0) == initial_scale * base(1.0)
    # End of warmup window: scaling factor has interpolated to 1.0.
    assert abs(schedule(1.0 - warmup_frac) - base(1.0 - warmup_frac)) < 1e-12
    # End of training: base schedule (unchanged).
    assert schedule(0.0) == base(0.0)
    # Past warmup window: base schedule applies.
    assert schedule(0.5) == base(0.5)


def test_shouldInterpolateScaleLinearlyAcrossWarmupWindow() -> None:
    # given
    base = _linear_schedule(1.0, 1.0)  # flat base so only the scale is visible
    schedule = _warmup_schedule(1.0, 1.0, warmup_frac=0.1, initial_scale=0.5)

    # when
    midpoint = schedule(0.95)  # halfway through the 0.1-wide warmup window

    # then
    # scale interpolates from 0.5 (at p=1.0) to 1.0 (at p=0.9), so midpoint = 0.75.
    assert abs(midpoint - 0.75 * base(0.95)) < 1e-12


def test_shouldOverrideHyperparamsFromConfigWhenLoadingBcCheckpoint(tmp_path: Path) -> None:
    # given — BC checkpoint saved via the trainer's transient MaskablePPO,
    # which pins ent_coef=0, n_epochs=1, n_steps=8, batch_size=8.
    obs_space = observation_space()
    act_space: spaces.Space[int] = spaces.Discrete(MAX_ACTIONS)
    policy = build_bc_policy(obs_space, act_space)
    ckpt_path = tmp_path / "bc.zip"
    save_bc_checkpoint(policy, obs_space, act_space, ckpt_path)
    vec_env = DummyVecEnv([lambda: _DummyEnv(obs_space, act_space)])
    cfg = TrainConfig(
        total_timesteps=1,
        character_id="jedrek",
        difficulty="normal",
        base_seed=None,
        checkpoint_dir=tmp_path / "run",
        engine_dir=None,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        ent_coef=0.01,
        n_epochs=10,
        gamma=0.97,
        gae_lambda=0.93,
        clip_range=0.1,
        bc_init=ckpt_path,
        initial_lr_scale=0.33,
        warmup_frac=0.1,
    )

    # when
    try:
        model = _load_bc_model(cfg, vec_env)

        # then — scalar hyperparams match the config, not the BC-transient defaults.
        assert model.ent_coef == 0.01
        assert model.n_epochs == 10
        assert model.n_steps == 1024
        assert model.batch_size == 64
        assert model.gamma == 0.97
        assert model.gae_lambda == 0.93
        # Rollout buffer was rebuilt to match n_steps/gamma/gae_lambda.
        assert model.rollout_buffer.buffer_size == 1024
        assert model.rollout_buffer.gamma == 0.97
        assert model.rollout_buffer.gae_lambda == 0.93
        assert model.rollout_buffer.n_envs == vec_env.num_envs
        # LR + clip-range overrides from custom_objects still applied.
        assert abs(model.lr_schedule(1.0) - 0.33 * 3e-4) < 1e-12
        clip_range_fn: Callable[[float], float] = model.clip_range  # type: ignore[assignment]
        assert clip_range_fn(1.0) == 0.1
    finally:
        vec_env.close()
