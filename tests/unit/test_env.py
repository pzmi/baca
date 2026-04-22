"""Unit tests for :mod:`baca.env` RPC wiring (observation mode + summary path)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

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


def test_shouldReturnOneWhenWinWithNoShaping() -> None:
    # given
    rpc = MagicMock()
    start_obs = _stub_observation(legal_actions=[{"type": "end_turn"}])
    done_obs = {**start_obs, "done": True, "outcome": "player_win", "floor": 15}
    rpc.call.side_effect = [
        {"runId": "r"},
        {"observation": start_obs},
        {"observation": done_obs, "summary": {}},
    ]
    env = UsiecCepraEnv(rpc, reward_shape="none")
    env.reset(seed=1)

    # when
    _obs, reward, done, _truncated, _info = env.step(0)

    # then
    assert done is True
    assert reward == 1.0


def test_shouldReturnOneWhenWinAtFloorFifteenWithFloorShaping() -> None:
    # given
    rpc = MagicMock()
    start_obs = _stub_observation(legal_actions=[{"type": "end_turn"}])
    done_obs = {**start_obs, "done": True, "outcome": "player_win", "floor": 15}
    rpc.call.side_effect = [
        {"runId": "r"},
        {"observation": start_obs},
        {"observation": done_obs, "summary": {}},
    ]
    env = UsiecCepraEnv(rpc, reward_shape="floor")
    env.reset(seed=1)

    # when
    _obs, reward, done, _truncated, _info = env.step(0)

    # then
    assert done is True
    assert reward == pytest.approx(1.0)


def test_shouldReturnTenHundredthsWhenLossAtFloorFifteenWithFloorShaping() -> None:
    # given
    rpc = MagicMock()
    start_obs = _stub_observation(legal_actions=[{"type": "end_turn"}])
    done_obs = {**start_obs, "done": True, "outcome": "enemy_win", "floor": 15}
    rpc.call.side_effect = [
        {"runId": "r"},
        {"observation": start_obs},
        {"observation": done_obs, "summary": {}},
    ]
    env = UsiecCepraEnv(rpc, reward_shape="floor")
    env.reset(seed=1)

    # when
    _obs, reward, done, _truncated, _info = env.step(0)

    # then
    assert done is True
    assert reward == pytest.approx(0.1)


def test_shouldScaleShapedRewardProportionallyWhenLossAtPartialFloor() -> None:
    # given
    rpc = MagicMock()
    start_obs = _stub_observation(legal_actions=[{"type": "end_turn"}])
    done_obs = {**start_obs, "done": True, "outcome": "enemy_win", "floor": 6}
    rpc.call.side_effect = [
        {"runId": "r"},
        {"observation": start_obs},
        {"observation": done_obs, "summary": {}},
    ]
    env = UsiecCepraEnv(rpc, reward_shape="floor")
    env.reset(seed=1)

    # when
    _obs, reward, _done, _truncated, _info = env.step(0)

    # then
    assert reward == pytest.approx(0.1 * (6 / 15))


def test_shouldReturnZeroWhenTruncatedRegardlessOfShaping() -> None:
    # given
    rpc = MagicMock()
    start_obs = _stub_observation(legal_actions=[{"type": "end_turn"}])
    # Not done, no outcome — step returns reward 0 and (if step_count exceeds
    # max) truncated True.
    continued_obs = {**start_obs, "done": False}
    rpc.call.side_effect = [
        {"runId": "r"},
        {"observation": start_obs},
        {"observation": continued_obs},
    ]
    env = UsiecCepraEnv(rpc, reward_shape="floor", max_episode_steps=1)
    env.reset(seed=1)

    # when
    _obs, reward, done, truncated, _info = env.step(0)

    # then
    assert done is False
    assert truncated is True
    assert reward == 0.0


def test_shouldIgnoreShapingWhenRewardShapeNone() -> None:
    # given
    rpc = MagicMock()
    start_obs = _stub_observation(legal_actions=[{"type": "end_turn"}])
    done_obs = {**start_obs, "done": True, "outcome": "enemy_win", "floor": 12}
    rpc.call.side_effect = [
        {"runId": "r"},
        {"observation": start_obs},
        {"observation": done_obs, "summary": {}},
    ]
    env = UsiecCepraEnv(rpc, reward_shape="none")
    env.reset(seed=1)

    # when
    _obs, reward, _done, _truncated, _info = env.step(0)

    # then
    assert reward == 0.0


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
