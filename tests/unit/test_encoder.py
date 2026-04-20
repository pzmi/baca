"""Unit tests for :mod:`baca.encoder`."""

from __future__ import annotations

import numpy as np

from baca.encoder import (
    CARD_TYPES,
    MAX_ACTIONS,
    MAX_HAND_SIZE,
    PHASES,
    PLAYER_STATUSES,
    encode,
    observation_space,
)


def _minimal_observation() -> dict[str, object]:
    return {
        "phase": "battle",
        "turn": 0,
        "battleTurn": 0,
        "floor": 2,
        "act": 1,
        "done": False,
        "weather": {"id": "clear", "name": "", "description": ""},
        "player": {
            "hp": 40,
            "maxHp": 60,
            "block": 5,
            "energy": 3,
            "maxEnergy": 3,
            "status": {"weak": {"amount": 2}},
            "stunned": False,
            "cardsPlayedThisTurn": 1,
        },
        "enemy": {
            "id": "cepr",
            "name": "Ceper",
            "hp": 20,
            "maxHp": 30,
            "block": 0,
            "status": {},
            "passive": None,
            "isElite": False,
            "isBoss": False,
            "rachunek": 0,
            "ped": 0,
            "intent": {"type": "attack", "expectedDamageToPlayer": 8},
            "phaseTwoTriggered": False,
        },
        "hand": [
            {"id": "ciupaga", "type": "attack", "cost": 1, "unplayable": False},
            {"id": "garda", "type": "skill", "cost": 1, "unplayable": False},
        ],
        "deckCount": 10,
        "discardCount": 2,
        "exhaustCount": 0,
        "combat": {"firstAttackUsed": False, "activeSide": "player"},
        "run": {
            "character": "jedrek",
            "difficulty": "normal",
            "dutki": 50,
            "relics": [],
            "marynaBoon": None,
        },
        "legalActions": [
            {"type": "play_card", "handIndex": 0},
            {"type": "play_card", "handIndex": 1},
            {"type": "end_turn"},
        ],
    }


def test_shouldProduceArraysMatchingObservationSpace() -> None:
    # given
    obs = _minimal_observation()

    # when
    encoded = encode(obs)

    # then
    space = observation_space()
    for key, sub_space in space.spaces.items():
        assert key in encoded, f"missing key {key}"
        assert encoded[key].shape == sub_space.shape, f"shape mismatch for {key}"


def test_shouldEncodePlayerHpRatioWhenBattleActive() -> None:
    # given
    obs = _minimal_observation()

    # when
    encoded = encode(obs)

    # then
    assert np.isclose(encoded["player"][0], 40 / 60)
    assert encoded["player"].shape == (4 + len(PLAYER_STATUSES),)


def test_shouldMarkEnemyPresentWhenEnemyInObservation() -> None:
    # given
    obs = _minimal_observation()

    # when
    encoded = encode(obs)

    # then
    assert encoded["enemy_present"][0] == 1.0
    assert np.isclose(encoded["enemy"][0], 20 / 30)


def test_shouldZeroEnemyFeaturesWhenOutOfBattle() -> None:
    # given
    obs = _minimal_observation()
    obs["enemy"] = None
    obs["phase"] = "map"

    # when
    encoded = encode(obs)

    # then
    assert encoded["enemy_present"][0] == 0.0
    assert np.all(encoded["enemy"] == 0.0)


def test_shouldOneHotPhaseWhenKnownPhase() -> None:
    # given
    obs = _minimal_observation()
    obs["phase"] = "shop"

    # when
    encoded = encode(obs)

    # then
    assert encoded["phase"].sum() == 1.0
    assert encoded["phase"][PHASES.index("shop")] == 1.0


def test_shouldMaskLegalActionsWhenProvided() -> None:
    # given
    obs = _minimal_observation()

    # when
    encoded = encode(obs)

    # then
    assert encoded["action_mask"].shape == (MAX_ACTIONS,)
    assert encoded["action_mask"][:3].sum() == 3
    assert encoded["action_mask"][3:].sum() == 0


def test_shouldPadHandWhenSmallerThanMaxSize() -> None:
    # given
    obs = _minimal_observation()

    # when
    encoded = encode(obs)

    # then
    assert encoded["hand"].shape == (MAX_HAND_SIZE, 2 + len(CARD_TYPES))
    assert encoded["hand_mask"][0] == 1
    assert encoded["hand_mask"][1] == 1
    assert encoded["hand_mask"][2] == 0
    # Playable attack card: playable=1, one-hot attack
    assert encoded["hand"][0, 1] == 1.0
    assert encoded["hand"][0, 2 + CARD_TYPES.index("attack")] == 1.0
