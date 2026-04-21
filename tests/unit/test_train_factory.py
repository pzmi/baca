"""Unit tests for :func:`baca.train.make_env` — the per-worker env thunk factory.

SubprocVecEnv spawns workers that each call a picklable env thunk. The thunk
must construct its own :class:`RpcClient` inside the worker (the client is not
picklable), and distinct ranks must produce distinct thunk instances so each
worker owns an independent subprocess.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import gymnasium
import numpy as np
from gymnasium import spaces

from baca import train as train_module


class _FakeEnv(gymnasium.Env[np.ndarray, int]):
    """Minimal concrete :class:`gymnasium.Env` so :class:`Monitor` accepts it."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        _ = seed, options
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        _ = action
        return np.zeros(1, dtype=np.float32), 0.0, True, False, {}


def test_shouldReturnDistinctCallablesWhenMultipleRanksRequested() -> None:
    # given / when
    thunk_a: Callable[[], Any] = train_module.make_env(
        rank=0,
        character_id="jedrek",
        difficulty="normal",
        base_seed=1,
        engine_dir=None,
    )
    thunk_b: Callable[[], Any] = train_module.make_env(
        rank=1,
        character_id="jedrek",
        difficulty="normal",
        base_seed=1,
        engine_dir=None,
    )

    # then
    assert thunk_a is not thunk_b
    assert callable(thunk_a)
    assert callable(thunk_b)


def test_shouldConstructIndependentRpcClientsWhenMultipleWorkersSpawn() -> None:
    # given
    thunks = [
        train_module.make_env(
            rank=i,
            character_id="jedrek",
            difficulty="normal",
            base_seed=7,
            engine_dir=None,
        )
        for i in range(3)
    ]

    # when
    with (
        patch.object(train_module, "RpcClient") as rpc_cls,
        patch.object(train_module, "UsiecCepraEnv") as env_cls,
    ):
        rpc_cls.side_effect = lambda **_: MagicMock()
        env_cls.side_effect = lambda *_a, **_kw: _FakeEnv()
        for thunk in thunks:
            thunk()

    # then
    assert rpc_cls.call_count == 3
    assert env_cls.call_count == 3


def test_shouldOffsetWorkerSeedWhenRankGreaterThanZero() -> None:
    # given
    thunk = train_module.make_env(
        rank=2,
        character_id="jedrek",
        difficulty="normal",
        base_seed=100,
        engine_dir=None,
    )

    # when
    with (
        patch.object(train_module, "RpcClient") as rpc_cls,
        patch.object(train_module, "UsiecCepraEnv") as env_cls,
    ):
        rpc_cls.return_value = MagicMock()
        env_cls.return_value = _FakeEnv()
        thunk()

    # then
    env_kwargs = env_cls.call_args.kwargs
    assert env_kwargs["base_seed"] == 100 + 2 * 1_000_000


def test_shouldPassNoneSeedWhenBaseSeedIsNone() -> None:
    # given
    thunk = train_module.make_env(
        rank=3,
        character_id="jedrek",
        difficulty="normal",
        base_seed=None,
        engine_dir=None,
    )

    # when
    with (
        patch.object(train_module, "RpcClient") as rpc_cls,
        patch.object(train_module, "UsiecCepraEnv") as env_cls,
    ):
        rpc_cls.return_value = MagicMock()
        env_cls.return_value = _FakeEnv()
        thunk()

    # then
    env_kwargs = env_cls.call_args.kwargs
    assert env_kwargs["base_seed"] is None
