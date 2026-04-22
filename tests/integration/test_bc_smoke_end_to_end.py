"""Integration smoke test: collect → BC train → eval on a real engine.

Composes Steps 1-4 at a subsampled scale (50 games / 2 epochs / 5 eval
episodes) so CI runs in < 15 min. The full 1k-game smoke is a manual
procedure documented in the implementation plan §5 Step 5.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from sb3_contrib import MaskablePPO

from baca.bc.generate_cli import run as generate_run
from baca.bc.train_cli import run as train_run
from baca.encoder import MAX_ACTIONS
from baca.env import UsiecCepraEnv
from baca.rpc_client import RpcClient


def _engine_dir() -> Path:
    override = os.environ.get("BACA_ENGINE_DIR")
    if override:
        return Path(override).resolve()
    return (Path(__file__).resolve().parent.parent.parent.parent / "slay-the-ceper").resolve()


def _engine_present() -> bool:
    return (_engine_dir() / "scripts" / "rpc-server.js").is_file()


def _node_present() -> bool:
    return shutil.which("node") is not None


def _run_eval_episodes(
    model: MaskablePPO, env: UsiecCepraEnv, n_episodes: int
) -> tuple[float, int]:
    floors: list[int] = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        max_floor = 0
        while not done and steps < 2000:
            mask = env.action_masks()
            # `.predict` wants batched obs; stack to shape (1, ...).
            batched = {k: np.expand_dims(v, axis=0) for k, v in obs.items()}
            action, _ = model.predict(
                batched, action_masks=np.expand_dims(mask, axis=0), deterministic=True
            )
            idx = int(action[0])
            assert mask[idx], f"predicted action {idx} is not legal; mask={mask.tolist()}"
            obs, _reward, terminated, truncated, _info = env.step(idx)
            done = terminated or truncated
            steps += 1
            cur_floor = int(env.unwrapped.__dict__.get("_last_obs", {}).get("floor") or 0)
            max_floor = max(max_floor, cur_floor)
        floors.append(max_floor)
    return float(np.mean(floors)), int(np.max(floors))


@pytest.mark.skipif(not _node_present(), reason="node binary not found")
@pytest.mark.skipif(not _engine_present(), reason="engine checkout missing")
@pytest.mark.timeout(900)
def test_shouldReachAverageFloorAboveThreeWhenTrainedOnSmokeDataset(tmp_path: Path) -> None:
    # given
    data_dir = tmp_path / "data"
    ckpt_path = tmp_path / "bc.zip"

    # when — collect 50 games from HeuristicBot
    collected = generate_run(
        out_dir=data_dir,
        games=50,
        base_seed=1,
        character_id="jedrek",
        difficulty="normal",
        engine_dir=_engine_dir(),
    )
    assert collected == 50

    # BC-train 2 epochs on them
    train_run(
        data_dir=data_dir,
        out_path=ckpt_path,
        n_epochs=2,
        batch_size=64,
        lr=1e-3,
        holdout_frac=0.1,
        seed=42,
        device="cpu",
    )
    assert ckpt_path.is_file()

    # Eval 5 episodes using the real engine env + loaded checkpoint.
    rpc = RpcClient(engine_dir=_engine_dir())
    try:
        env = UsiecCepraEnv(rpc, base_seed=99999)
        try:
            model = MaskablePPO.load(str(ckpt_path), env=None)
            avg_floor, max_floor = _run_eval_episodes(model, env, n_episodes=5)
        finally:
            env.close()
    finally:
        rpc.close()

    # then
    assert avg_floor >= 3.0, f"avg_floor={avg_floor:.2f} max_floor={max_floor}"
    assert max_floor <= 15  # sanity — engine caps at floor 15.
    # Sanity on the policy's action space shape.
    assert model.action_space.n == MAX_ACTIONS  # type: ignore[attr-defined]
