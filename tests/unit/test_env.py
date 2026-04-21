"""Unit tests for :mod:`baca.env` RPC wiring (observation mode + summary path)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from baca.env import UsiecCepraEnv


def _stub_observation(legal_actions: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a minimal observation payload accepted by :func:`baca.encoder.encode`."""
    return {
        "phase": "battle",
        "turn": 1,
        "battleTurn": 1,
        "floor": 1,
        "act": 1,
        "done": False,
        "player": {
            "hp": 60,
            "maxHp": 60,
            "block": 0,
            "energy": 3,
            "maxEnergy": 3,
            "status": {},
            "cardsPlayedThisTurn": 0,
        },
        "enemy": None,
        "hand": [],
        "deckCount": 10,
        "discardCount": 0,
        "exhaustCount": 0,
        "run": {"dutki": 0},
        "legalActions": legal_actions,
    }


def test_shouldPassAgentObservationModeWhenStartingRun() -> None:
    # given
    rpc = MagicMock()
    obs = _stub_observation(legal_actions=[{"type": "end_turn"}])
    rpc.call.side_effect = [
        {"runId": "r-1"},
        {"observation": obs},
    ]
    env = UsiecCepraEnv(rpc, character_id="jedrek", difficulty="normal", base_seed=1)

    # when
    env.reset(seed=42)

    # then
    create_call = rpc.call.call_args_list[0]
    assert create_call.args[0] == "engine.create"
    create_params = create_call.args[1]
    assert create_params["rules"]["observationMode"] == "agent"
    assert create_params["rules"]["revealAllPiles"] is False


def test_shouldPreferInlineSummaryWhenTerminalResultHasIt() -> None:
    # given
    rpc = MagicMock()
    start_obs = _stub_observation(legal_actions=[{"type": "end_turn"}])
    done_obs = {**start_obs, "done": True, "outcome": "player_win"}
    inline_summary = {"floorReached": 9, "victory": True}
    rpc.call.side_effect = [
        {"runId": "r-2"},
        {"observation": start_obs},
        {"observation": done_obs, "summary": inline_summary},
    ]
    env = UsiecCepraEnv(rpc)
    env.reset(seed=1)

    # when
    _obs, _reward, done, _truncated, info = env.step(0)

    # then
    assert done is True
    assert info["summary"] == inline_summary
    methods = [call.args[0] for call in rpc.call.call_args_list]
    assert "engine.getRunSummary" not in methods


def test_shouldFallBackToGetRunSummaryWhenInlineSummaryMissing() -> None:
    # given
    rpc = MagicMock()
    start_obs = _stub_observation(legal_actions=[{"type": "end_turn"}])
    done_obs = {**start_obs, "done": True, "outcome": "enemy_win"}
    fallback_summary = {"floorReached": 3, "victory": False}
    rpc.call.side_effect = [
        {"runId": "r-3"},
        {"observation": start_obs},
        {"observation": done_obs},
        {"summary": fallback_summary},
    ]
    env = UsiecCepraEnv(rpc)
    env.reset(seed=1)

    # when
    _obs, _reward, done, _truncated, info = env.step(0)

    # then
    assert done is True
    assert info["summary"] == fallback_summary
    methods = [call.args[0] for call in rpc.call.call_args_list]
    assert methods.count("engine.getRunSummary") == 1
