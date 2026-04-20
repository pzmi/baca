"""End-to-end smoke test: spawn the engine, play one random episode."""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

from baca.env import UsiecCepraEnv
from baca.rpc_client import RpcClient


def _engine_dir() -> Path:
    override = os.environ.get("BACA_ENGINE_DIR")
    if override:
        return Path(override).resolve()
    return (Path(__file__).resolve().parent.parent.parent.parent / "slay-the-ceper").resolve()


@pytest.mark.skipif(
    not (_engine_dir() / "scripts" / "rpc-server.js").is_file(),
    reason="slay-the-ceper engine checkout not found next to BACA",
)
def test_shouldFinishEpisodeWhenPlayingRandomActions() -> None:
    # given
    rpc = RpcClient(engine_dir=_engine_dir())
    rng = random.Random(12345)

    # when
    try:
        env = UsiecCepraEnv(rpc, base_seed=1)
        _obs, info = env.reset()
        assert "runId" in info

        done = False
        steps = 0
        while not done and steps < 2000:
            mask = env.action_masks()
            legal_indices = [i for i, allowed in enumerate(mask) if allowed]
            assert legal_indices, "legal action set must be non-empty until done"
            action = rng.choice(legal_indices)
            _obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        # then
        assert done, "episode should terminate within 2000 steps"
        assert info.get("outcome") in {"player_win", "enemy_win"}
        env.close()
    finally:
        rpc.close()
