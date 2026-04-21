"""Unit tests for :mod:`baca.encoder`."""

from __future__ import annotations

import numpy as np

from baca.encoder import (
    _CARD_TYPE_INDEX,
    _PHASE_INDEX,
    _PLAYER_STATUS_INDEX,
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


def test_shouldReturnCorrectIndexWhenPhaseKnown() -> None:
    # given
    phases = PHASES

    # when / then
    for i, phase in enumerate(phases):
        assert _PHASE_INDEX[phase] == i
        assert _PHASE_INDEX[phase] == PHASES.index(phase)
    assert len(_PHASE_INDEX) == len(PHASES)


def test_shouldReturnCorrectIndexWhenCardTypeKnown() -> None:
    # given
    types = CARD_TYPES

    # when / then
    for i, card_type in enumerate(types):
        assert _CARD_TYPE_INDEX[card_type] == i
        assert _CARD_TYPE_INDEX[card_type] == CARD_TYPES.index(card_type)
    assert len(_CARD_TYPE_INDEX) == len(CARD_TYPES)


def test_shouldReturnCorrectIndexWhenPlayerStatusKnown() -> None:
    # given
    statuses = PLAYER_STATUSES

    # when / then
    for i, status in enumerate(statuses):
        assert _PLAYER_STATUS_INDEX[status] == i
        assert _PLAYER_STATUS_INDEX[status] == PLAYER_STATUSES.index(status)
    assert len(_PLAYER_STATUS_INDEX) == len(PLAYER_STATUSES)


def test_shouldReturnMinusOneWhenPhaseUnknown() -> None:
    # given
    unknown_phase = "not_a_real_phase"

    # when
    idx = _PHASE_INDEX.get(unknown_phase, -1)

    # then
    assert idx == -1


def test_shouldProduceIndependentOutputsWhenEncodedConsecutively() -> None:
    # given
    obs_a = _minimal_observation()
    obs_b = _minimal_observation()
    obs_b["phase"] = "shop"
    obs_b["enemy"] = None

    # when
    encoded_a_first = encode(obs_a)
    encoded_b = encode(obs_b)
    encoded_a_second = encode(obs_a)

    # then — encoding obs_b must not mutate the arrays previously returned for obs_a
    for key in encoded_a_first:
        np.testing.assert_array_equal(encoded_a_first[key], encoded_a_second[key])
    assert encoded_b["enemy_present"][0] == 0.0
    assert encoded_a_first["enemy_present"][0] == 1.0
