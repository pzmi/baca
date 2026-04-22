"""Integration smoke test for BC dataset collection CLI."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from baca.bc.generate_cli import MANIFEST_NAME, SNAPSHOT_NAME
from baca.bc.generate_cli import main as generate_main
from baca.encoder import MAX_ACTIONS


def _engine_dir() -> Path:
    override = os.environ.get("BACA_ENGINE_DIR")
    if override:
        return Path(override).resolve()
    return (Path(__file__).resolve().parent.parent.parent.parent / "slay-the-ceper").resolve()


def _engine_present() -> bool:
    return (_engine_dir() / "scripts" / "rpc-server.js").is_file()


def _node_present() -> bool:
    return shutil.which("node") is not None


@pytest.mark.skipif(not _node_present(), reason="node binary not found")
@pytest.mark.skipif(not _engine_present(), reason="engine checkout missing")
@pytest.mark.timeout(180)
def test_shouldCollectFiveGamesWhenDrivenByHeuristicBot(tmp_path: Path) -> None:
    # given
    out_dir = tmp_path / "v3-smoke"

    # when
    rc = generate_main(
        [
            "--out",
            str(out_dir),
            "--games",
            "5",
            "--base-seed",
            "100",
            "--engine-dir",
            str(_engine_dir()),
        ]
    )

    # then
    assert rc == 0
    assert (out_dir / SNAPSHOT_NAME).is_file()
    manifest = json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert isinstance(manifest, list)
    assert len(manifest) == 5
    npz_files = sorted(out_dir.glob("seed_*.npz"))
    assert len(npz_files) == 5
    for path in npz_files:
        with np.load(path) as data:
            actions = np.asarray(data["action_indices"])
            assert actions.dtype == np.int64
            assert actions.ndim == 1
            if actions.shape[0] > 0:
                assert int(actions.min()) >= 0
                assert int(actions.max()) < MAX_ACTIONS
