"""Observation encoder: engine JSON → numpy arrays for a neural policy.

Design goals: small fixed-size feature vector + padded hand + legal-action
mask, plus per-card identity ids for hand / shop / reward blocks that feed an
embedding-backed policy (see :mod:`baca.policy`).

Checkpoint stability contract: ``PHASES``, ``CARD_TYPES``, ``PLAYER_STATUSES``
and ``CARD_IDS`` are ordered APPEND-ONLY tuples. New entries must be appended
at the tail — inserting or reordering shifts indices in saved embeddings and
breaks every previously trained policy. New engine cards that are not yet in
``CARD_IDS`` map to the reserved ``UNKNOWN`` bucket at index 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

# Canonical phase ordering; any future phase must be appended at the end to
# preserve encoder stability for saved checkpoints.
PHASES: tuple[str, ...] = (
    "battle",
    "map",
    "reward",
    "shop",
    "campfire",
    "event",
    "maryna",
)

CARD_TYPES: tuple[str, ...] = ("attack", "skill", "power", "status", "curse")

# Ordered and APPEND-ONLY. Index 0 is reserved for unknown cards so that new
# engine-side cards map to the UNKNOWN bucket instead of shifting indices.
# Seeded in engine order from slay-the-ceper/src/data/cards.js.
CARD_IDS: tuple[str, ...] = (
    "UNKNOWN",
    "ciupaga",
    "gasior",
    "goralska_obrona",
    "kierpce",
    "hej",
    "sernik",
    "redyk",
    "halny",
    "paragon_za_gofra",
    "podatek_klimatyczny",
    "wypozyczone_gogle",
    "zdjecie_z_misiem",
    "parzenica",
    "zadyma",
    "zyntyca",
    "janosik",
    "echo",
    "sandaly",
    "giewont",
    "mocny_organizm",
    "pchniecie_ciupaga",
    "barchanowe_gacie",
    "szukanie_okazji",
    "lodolamacz",
    "duma_podhala",
    "zemsta_gorala",
    "prestiz_na_kredyt",
    "furia_turysty",
    "spostrzegawczosc",
    "pocieszenie",
    "ulotka",
    "spam_tagami",
    "wydruk_z_kasy",
    "nadplacony_bilet",
    "eksmisja_z_kwatery",
    "rachunek_za_oddychanie",
    "skrupulatne_wyliczenie",
    "tatrzanski_szpan",
    "paradny_zwyrt",
    "cios_z_telemarkiem",
    "mlynek_ciupaga",
    "wepchniecie_w_kolejke",
    "rozped_z_rowni",
    "z_rozmachu",
    "beczenie_redyku",
    "krzesany",
    "wymuszony_napiwek",
    "paragon_grozy",
    "pogodzenie_sporow",
    "przymusowy_napiwek",
    "list_od_maryny",
    "zapas_oscypkow",
    "wdech_halnego",
    "dutki_na_stole",
    "pan_na_wlosciach",
    "zimna_krew",
    "baciarka_ciesy",
    "czas_na_fajke",
    "goralska_goscinnosc",
    "koncesja_na_oscypki",
    "ciupaga_we_mgle",
    "przymusowe_morsowanie",
    "lawina_z_morskiego_oka",
    "punkt_widokowy",
    "zgubieni_we_mgle",
    "znajomosc_szlaku",
    "kapiel_w_bialce",
)
CARD_VOCAB_SIZE = len(CARD_IDS)

MAX_HAND_SIZE = 10
MAX_ACTIONS = 32
# Structural caps on phase-conditional card lists. Both are hard-coded in the
# engine today (ShopSystem.generateShopStock + ActionDispatcher reward offer
# both call _pickUniqueItems(..., 3)); widening here would require engine
# changes first and invalidates saved policies.
MAX_REWARD_CARDS = 3
MAX_SHOP_CARDS = 3
PLAYER_STATUSES: tuple[str, ...] = (
    "weak",
    "vulnerable",
    "fragile",
    "strength",
    "lans",
    "duma_podhala",
    "furia_turysty",
    "next_double",
)

# O(1) lookup dicts mirroring the tuples above. Must stay in sync; append-only
# semantics on the tuples carry over here.
_PHASE_INDEX: dict[str, int] = {name: i for i, name in enumerate(PHASES)}
_CARD_TYPE_INDEX: dict[str, int] = {name: i for i, name in enumerate(CARD_TYPES)}
_PLAYER_STATUS_INDEX: dict[str, int] = {name: i for i, name in enumerate(PLAYER_STATUSES)}
_CARD_ID_INDEX: dict[str, int] = {name: i for i, name in enumerate(CARD_IDS)}


def observation_space() -> spaces.Dict:
    """Gymnasium observation space matching :func:`encode` output."""
    max_id = max(0, CARD_VOCAB_SIZE - 1)
    return spaces.Dict(
        {
            "player": spaces.Box(
                low=-1.0, high=1.0, shape=(4 + len(PLAYER_STATUSES),), dtype=np.float32
            ),
            "enemy": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
            "enemy_present": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "phase": spaces.Box(low=0.0, high=1.0, shape=(len(PHASES),), dtype=np.float32),
            "run": spaces.Box(low=0.0, high=2.0, shape=(4,), dtype=np.float32),
            "hand": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(MAX_HAND_SIZE, 2 + len(CARD_TYPES)),
                dtype=np.float32,
            ),
            "hand_mask": spaces.MultiBinary(MAX_HAND_SIZE),
            "hand_ids": spaces.Box(low=0, high=max_id, shape=(MAX_HAND_SIZE,), dtype=np.int64),
            "reward_ids": spaces.Box(low=0, high=max_id, shape=(MAX_REWARD_CARDS,), dtype=np.int64),
            "reward_mask": spaces.MultiBinary(MAX_REWARD_CARDS),
            "reward_costs": spaces.Box(
                low=0.0, high=1.0, shape=(MAX_REWARD_CARDS,), dtype=np.float32
            ),
            "shop_ids": spaces.Box(low=0, high=max_id, shape=(MAX_SHOP_CARDS,), dtype=np.int64),
            "shop_mask": spaces.MultiBinary(MAX_SHOP_CARDS),
            "shop_costs": spaces.Box(low=0.0, high=1.0, shape=(MAX_SHOP_CARDS,), dtype=np.float32),
            "action_mask": spaces.MultiBinary(MAX_ACTIONS),
        }
    )


def encode(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Encode one engine observation into a dict of numpy arrays."""
    hand = obs.get("hand") or []
    reward_ids, reward_mask, reward_costs = _encode_reward_cards(obs.get("rewardOffer"))
    shop_ids, shop_mask, shop_costs = _encode_shop_cards(obs.get("shopStock"))
    return {
        "player": _encode_player(obs["player"]),
        "enemy": _encode_enemy(obs.get("enemy"), obs["player"]["maxHp"]),
        "enemy_present": np.array([1.0 if obs.get("enemy") else 0.0], dtype=np.float32),
        "phase": _encode_phase(obs["phase"]),
        "run": _encode_run(obs),
        "hand": _encode_hand(hand),
        "hand_mask": _encode_hand_mask(hand),
        "hand_ids": _encode_hand_ids(hand),
        "reward_ids": reward_ids,
        "reward_mask": reward_mask,
        "reward_costs": reward_costs,
        "shop_ids": shop_ids,
        "shop_mask": shop_mask,
        "shop_costs": shop_costs,
        "action_mask": _encode_action_mask(obs.get("legalActions") or []),
    }


def _encode_player(player: dict[str, Any]) -> np.ndarray:
    max_hp = max(1, int(player["maxHp"]))
    max_energy = max(1, int(player.get("maxEnergy", 1)))
    base = [
        player["hp"] / max_hp,
        min(1.0, player["block"] / 50.0),
        player["energy"] / max_energy,
        min(1.0, player.get("cardsPlayedThisTurn", 0) / 10.0),
    ]
    status = player.get("status") or {}
    base.extend(_status_magnitude(status, key) for key in PLAYER_STATUSES)
    return np.asarray(base, dtype=np.float32)


def _encode_enemy(enemy: dict[str, Any] | None, player_max_hp: int) -> np.ndarray:
    if enemy is None:
        return np.zeros(5, dtype=np.float32)
    max_hp = max(1, int(enemy.get("maxHp", 1)))
    intent = enemy.get("intent") or {}
    return np.asarray(
        [
            enemy["hp"] / max_hp,
            min(1.0, enemy.get("block", 0) / 50.0),
            min(1.0, (intent.get("expectedDamageToPlayer") or 0) / max(1, player_max_hp)),
            1.0 if enemy.get("isElite") else 0.0,
            1.0 if enemy.get("isBoss") else 0.0,
        ],
        dtype=np.float32,
    )


def _encode_phase(phase: str) -> np.ndarray:
    out = np.zeros(len(PHASES), dtype=np.float32)
    idx = _PHASE_INDEX.get(phase, -1)
    if idx >= 0:
        out[idx] = 1.0
    return out


def _encode_run(obs: dict[str, Any]) -> np.ndarray:
    run = obs.get("run") or {}
    return np.asarray(
        [
            min(2.0, obs.get("floor", 0) / 15.0),
            min(2.0, obs.get("act", 1) / 3.0),
            min(2.0, run.get("dutki", 0) / 200.0),
            min(2.0, obs.get("deckCount", 0) / 30.0),
        ],
        dtype=np.float32,
    )


def _encode_hand(hand: list[dict[str, Any]]) -> np.ndarray:
    out = np.zeros((MAX_HAND_SIZE, 2 + len(CARD_TYPES)), dtype=np.float32)
    for i, card in enumerate(hand[:MAX_HAND_SIZE]):
        cost = _normalize_cost(card.get("effectiveCost", card.get("cost", 0)))
        out[i, 0] = cost
        out[i, 1] = 0.0 if card.get("unplayable") else 1.0
        card_type = card.get("type")
        card_type_idx = _CARD_TYPE_INDEX.get(card_type, -1) if isinstance(card_type, str) else -1
        if card_type_idx >= 0:
            out[i, 2 + card_type_idx] = 1.0
    return out


def _encode_hand_mask(hand: list[dict[str, Any]]) -> np.ndarray:
    mask = np.zeros(MAX_HAND_SIZE, dtype=np.int8)
    mask[: min(len(hand), MAX_HAND_SIZE)] = 1
    return mask


def _encode_hand_ids(hand: list[dict[str, Any]]) -> np.ndarray:
    ids = np.zeros(MAX_HAND_SIZE, dtype=np.int64)
    for i, card in enumerate(hand[:MAX_HAND_SIZE]):
        ids[i] = _lookup_card_index(card.get("id"))
    return ids


def _encode_reward_cards(
    reward_offer: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.zeros(MAX_REWARD_CARDS, dtype=np.int64)
    mask = np.zeros(MAX_REWARD_CARDS, dtype=np.int8)
    costs = np.zeros(MAX_REWARD_CARDS, dtype=np.float32)
    if not reward_offer:
        return ids, mask, costs
    cards = reward_offer.get("cards") or []
    for i, entry in enumerate(cards[:MAX_REWARD_CARDS]):
        ids[i] = _lookup_card_index(_coerce_card_id(entry))
        mask[i] = 1
    return ids, mask, costs


def _encode_shop_cards(
    shop_stock: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.zeros(MAX_SHOP_CARDS, dtype=np.int64)
    mask = np.zeros(MAX_SHOP_CARDS, dtype=np.int8)
    costs = np.zeros(MAX_SHOP_CARDS, dtype=np.float32)
    if not shop_stock:
        return ids, mask, costs
    cards = shop_stock.get("cards") or []
    for i, entry in enumerate(cards[:MAX_SHOP_CARDS]):
        ids[i] = _lookup_card_index(_coerce_card_id(entry))
        mask[i] = 1
    return ids, mask, costs


def _lookup_card_index(card_id: Any) -> int:
    if not isinstance(card_id, str):
        return 0
    return _CARD_ID_INDEX.get(card_id, 0)


def _coerce_card_id(entry: Any) -> str | None:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        candidate = entry.get("id")
        if isinstance(candidate, str):
            return candidate
    return None


def _encode_action_mask(legal_actions: list[dict[str, Any]]) -> np.ndarray:
    mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
    mask[: min(len(legal_actions), MAX_ACTIONS)] = 1
    return mask


def _status_magnitude(status: dict[str, Any], key: str) -> float:
    entry = status.get(key)
    if entry is None:
        return 0.0
    if isinstance(entry, dict):
        amount = entry.get("amount", entry.get("value", entry.get("duration", 0))) or 0
        return min(1.0, float(amount) / 5.0)
    if isinstance(entry, (int, float, bool)):
        return min(1.0, float(entry) / 5.0)
    return 0.0


def _normalize_cost(cost: Any) -> float:
    if cost == "X" or cost == -1:
        return 1.0
    try:
        return min(1.0, max(0.0, float(cost) / 3.0))
    except TypeError, ValueError:
        return 0.0
