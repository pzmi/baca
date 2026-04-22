"""Unit tests for ``HeuristicBridge``."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import pytest

from baca.bc.heuristic_bridge import HeuristicBridge


def _engine_dir() -> Path:
    override = os.environ.get("BACA_ENGINE_DIR")
    if override:
        return Path(override).resolve()
    return (Path(__file__).resolve().parent.parent.parent.parent / "slay-the-ceper").resolve()


def _engine_present() -> bool:
    return (_engine_dir() / "src" / "logic" / "bots" / "HeuristicBot.js").is_file()


def _node_present() -> bool:
    return shutil.which("node") is not None


def _canonical(action: Any) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def _minimal_battle_observation(
    legal_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a minimal battle-phase observation that HeuristicBot can score.

    Mirrors the fields HeuristicBot touches — ``phase``, ``player``, ``enemy``,
    ``hand``, ``legalActions``. Missing optional fields are defaulted to empty.
    """
    return {
        "phase": "battle",
        "player": {
            "hp": 60,
            "maxHp": 70,
            "block": 0,
            "energy": 3,
            "status": {},
        },
        "enemy": {
            "hp": 30,
            "maxHp": 30,
            "block": 0,
            "rachunek": 0,
            "rachunekImmune": False,
            "status": {},
            "intent": {"expectedDamageToPlayer": 6},
        },
        "hand": [
            {
                "id": "strike",
                "cardId": "strike",
                "type": "attack",
                "effectiveCost": 1,
                "desc": "Zadaj 6 obrażeń.",
                "tags": [],
                "exhaust": False,
                "unplayable": False,
            },
        ],
        "legalActions": legal_actions,
    }


@pytest.mark.skipif(not _node_present(), reason="node binary not found")
@pytest.mark.skipif(not _engine_present(), reason="engine checkout missing")
def test_shouldReturnLegalActionWhenObservationHasSingleLegalAction() -> None:
    # given
    only = {"type": "end_turn"}
    obs = _minimal_battle_observation([only])

    # when
    with HeuristicBridge(engine_dir=_engine_dir()) as bridge:
        action = bridge.decide(obs)

    # then
    assert _canonical(action) == _canonical(only)


@pytest.mark.skipif(not _node_present(), reason="node binary not found")
@pytest.mark.skipif(not _engine_present(), reason="engine checkout missing")
def test_shouldReturnActionFromLegalListWhenObservationHasMultiple() -> None:
    # given
    end_turn: dict[str, Any] = {"type": "end_turn"}
    play_strike: dict[str, Any] = {"type": "play_card", "handIndex": 0}
    legal: list[dict[str, Any]] = [end_turn, play_strike]
    obs = _minimal_battle_observation(legal)

    # when
    with HeuristicBridge(engine_dir=_engine_dir()) as bridge:
        action = bridge.decide(obs)

    # then
    canonical_action = _canonical(action)
    assert canonical_action in {_canonical(a) for a in legal}


@pytest.mark.skipif(not _node_present(), reason="node binary not found")
@pytest.mark.skipif(not _engine_present(), reason="engine checkout missing")
def test_shouldRaiseWhenSubprocessDies() -> None:
    # given
    bridge = HeuristicBridge(engine_dir=_engine_dir())
    try:
        # when
        bridge._process.kill()
        bridge._process.wait(timeout=2.0)
        time.sleep(0.1)

        # then
        with pytest.raises(RuntimeError):
            bridge.decide(_minimal_battle_observation([{"type": "end_turn"}]))
    finally:
        bridge.close()
