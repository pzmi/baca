"""Unit tests for :mod:`baca.eval_cli` metrics aggregation."""

from __future__ import annotations

from typing import Any

from baca.eval_cli import EpisodeCollector, _aggregate


def _step_locals(done: bool, info: dict[str, Any]) -> dict[str, Any]:
    return {"done": done, "info": info}


def test_shouldIgnoreMidEpisodeStepsWhenDoneIsFalse() -> None:
    # given
    collector = EpisodeCollector(total=0)

    # when
    collector(_step_locals(done=False, info={"outcome": "player_win"}), {})

    # then
    assert collector.floors == []
    assert collector.wins == []


def test_shouldRecordWinAndFloorWhenEpisodeEndsInPlayerWin() -> None:
    # given
    collector = EpisodeCollector(total=0)

    # when
    collector(
        _step_locals(
            done=True,
            info={"outcome": "player_win", "summary": {"floorReached": 15}},
        ),
        {},
    )

    # then
    assert collector.wins == [1]
    assert collector.floors == [15]


def test_shouldRecordLossAndFloorWhenEpisodeEndsInEnemyWin() -> None:
    # given
    collector = EpisodeCollector(total=0)

    # when
    collector(
        _step_locals(
            done=True,
            info={"outcome": "enemy_win", "summary": {"floorReached": 7}},
        ),
        {},
    )

    # then
    assert collector.wins == [0]
    assert collector.floors == [7]


def test_shouldRecordZeroFloorWhenSummaryMissing() -> None:
    # given
    collector = EpisodeCollector(total=0)

    # when
    collector(_step_locals(done=True, info={"outcome": "enemy_win"}), {})

    # then
    assert collector.wins == [0]
    assert collector.floors == [0]


def test_shouldAggregateWinrateWhenMixedOutcomes() -> None:
    # given
    floors = [3, 10, 5, 12]
    wins = [0, 1, 0, 1]

    # when
    metrics = _aggregate(episodes=4, floors=floors, wins=wins)

    # then
    assert metrics == {
        "episodes": 4.0,
        "winrate": 0.5,
        "avg_floor": (3 + 10 + 5 + 12) / 4,
        "max_floor": 12.0,
    }


def test_shouldReportZeroFloorsWhenNoEpisodesFinished() -> None:
    # given
    floors: list[int] = []
    wins: list[int] = []

    # when
    metrics = _aggregate(episodes=0, floors=floors, wins=wins)

    # then
    assert metrics == {
        "episodes": 0.0,
        "winrate": 0.0,
        "avg_floor": 0.0,
        "max_floor": 0.0,
    }
