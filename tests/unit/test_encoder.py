"""Unit tests for :mod:`baca.encoder`."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest

from baca.encoder import (
    _CARD_ID_INDEX,
    _CARD_TYPE_INDEX,
    _PHASE_INDEX,
    _PLAYER_STATUS_INDEX,
    CARD_IDS,
    CARD_TYPES,
    CARD_VOCAB_SIZE,
    MAX_ACTIONS,
    MAX_HAND_SIZE,
    MAX_REWARD_CARDS,
    MAX_SHOP_CARDS,
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


def test_shouldIncludeUnknownSentinelWhenCardIdsInitialized() -> None:
    # given / when
    sentinel_index = _CARD_ID_INDEX["UNKNOWN"]

    # then
    assert CARD_IDS[0] == "UNKNOWN"
    assert sentinel_index == 0
    assert len(CARD_IDS) == CARD_VOCAB_SIZE
    assert len(set(CARD_IDS)) == len(CARD_IDS), "CARD_IDS must be unique"


def _engine_cards_path() -> Path | None:
    override = os.environ.get("BACA_ENGINE_DIR")
    if override:
        base = Path(override)
    else:
        base = Path(__file__).resolve().parent.parent.parent.parent / "slay-the-ceper"
    candidate = (base / "src" / "data" / "cards.js").resolve()
    return candidate if candidate.is_file() else None


@pytest.mark.skipif(_engine_cards_path() is None, reason="engine cards.js not found")
def test_shouldMatchEngineCardsFileWhenAuditCompared() -> None:
    # given
    cards_path = _engine_cards_path()
    assert cards_path is not None
    source = cards_path.read_text(encoding="utf-8")

    # when
    engine_ids = re.findall(r"^\s*id:\s*'([^']+)'", source, flags=re.MULTILINE)

    # then
    expected = ("UNKNOWN", *engine_ids)
    assert expected == CARD_IDS, (
        "CARD_IDS drifted from engine's cards.js. Append new engine ids to the "
        "tail of CARD_IDS; never insert or reorder (breaks saved checkpoints)."
    )


def test_shouldReturnCorrectIndexWhenCardIdKnown() -> None:
    # given
    ids = CARD_IDS

    # when / then
    for i, card_id in enumerate(ids):
        assert _CARD_ID_INDEX[card_id] == i
    assert len(_CARD_ID_INDEX) == len(CARD_IDS)


def test_shouldProduceHandIdsMatchingHandWhenCardsPresent() -> None:
    # given
    obs = _minimal_observation()

    # when
    encoded = encode(obs)

    # then
    assert encoded["hand_ids"].shape == (MAX_HAND_SIZE,)
    assert encoded["hand_ids"].dtype == np.int64
    assert encoded["hand_ids"][0] == _CARD_ID_INDEX["ciupaga"]
    # "garda" is not in CARD_IDS so should map to UNKNOWN sentinel at index 0
    assert encoded["hand_ids"][1] == 0
    assert encoded["hand_ids"][2] == 0  # padding


def test_shouldZeroFillRewardIdsWhenPhaseNotReward() -> None:
    # given
    obs = _minimal_observation()  # phase is "battle"

    # when
    encoded = encode(obs)

    # then
    assert encoded["reward_ids"].shape == (MAX_REWARD_CARDS,)
    assert encoded["reward_mask"].shape == (MAX_REWARD_CARDS,)
    assert encoded["reward_costs"].shape == (MAX_REWARD_CARDS,)
    assert np.all(encoded["reward_ids"] == 0)
    assert np.all(encoded["reward_mask"] == 0)
    assert np.all(encoded["reward_costs"] == 0.0)


def test_shouldEncodeRewardCardIdsWhenOfferPresent() -> None:
    # given
    obs = _minimal_observation()
    obs["phase"] = "reward"
    obs["enemy"] = None
    obs["rewardOffer"] = {"cards": ["ciupaga", "janosik"], "relicId": "bilet_tpn"}

    # when
    encoded = encode(obs)

    # then
    assert encoded["reward_ids"][0] == _CARD_ID_INDEX["ciupaga"]
    assert encoded["reward_ids"][1] == _CARD_ID_INDEX["janosik"]
    assert encoded["reward_ids"][2] == 0
    assert encoded["reward_mask"][0] == 1
    assert encoded["reward_mask"][1] == 1
    assert encoded["reward_mask"][2] == 0


def test_shouldEncodeShopCardIdsWhenStockPresent() -> None:
    # given
    obs = _minimal_observation()
    obs["phase"] = "shop"
    obs["enemy"] = None
    obs["shopStock"] = {"cards": ["giewont", "sandaly", "echo"], "relic": None}

    # when
    encoded = encode(obs)

    # then
    assert encoded["shop_ids"].shape == (MAX_SHOP_CARDS,)
    assert encoded["shop_ids"][0] == _CARD_ID_INDEX["giewont"]
    assert encoded["shop_ids"][1] == _CARD_ID_INDEX["sandaly"]
    assert encoded["shop_ids"][2] == _CARD_ID_INDEX["echo"]
    assert np.all(encoded["shop_mask"] == 1)


def test_shouldMapUnknownCardToIndexZeroWhenIdMissing() -> None:
    # given
    obs = _minimal_observation()
    obs["phase"] = "reward"
    obs["enemy"] = None
    obs["rewardOffer"] = {"cards": ["card_not_in_registry"], "relicId": None}

    # when
    encoded = encode(obs)

    # then
    assert encoded["reward_ids"][0] == 0  # UNKNOWN sentinel
    assert encoded["reward_mask"][0] == 1


def test_shouldRespectObservationSpaceDtypesWhenEncodingMixedFeatures() -> None:
    # given
    obs = _minimal_observation()

    # when
    encoded = encode(obs)

    # then
    assert encoded["hand_ids"].dtype == np.int64
    assert encoded["reward_ids"].dtype == np.int64
    assert encoded["shop_ids"].dtype == np.int64
    assert encoded["player"].dtype == np.float32
    assert encoded["hand"].dtype == np.float32
    assert encoded["reward_costs"].dtype == np.float32
    assert encoded["shop_costs"].dtype == np.float32


def test_shouldMatchObservationSpaceWhenAllNewKeysPresent() -> None:
    # given
    obs = _minimal_observation()
    obs["phase"] = "shop"
    obs["enemy"] = None
    obs["shopStock"] = {"cards": ["giewont", "sandaly"], "relic": "bilet_tpn"}

    # when
    encoded = encode(obs)

    # then
    space = observation_space()
    assert set(space.spaces.keys()) == set(encoded.keys())
    for key, sub_space in space.spaces.items():
        assert encoded[key].shape == sub_space.shape, f"shape mismatch for {key}"


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
