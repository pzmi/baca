"""Observation encoder: engine JSON → numpy arrays for a neural policy.

Design goals for v0: small fixed-size feature vector + padded hand + legal-action
mask. No card-identity embedding yet (cards are encoded by numeric attributes).
This is deliberately minimal — card-id embeddings and history stacking are
deferred to a later phase.
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

MAX_HAND_SIZE = 10
MAX_ACTIONS = 32
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


def observation_space() -> spaces.Dict:
    """Gymnasium observation space matching :func:`encode` output."""
    return spaces.Dict(
        {
            "player": spaces.Box(low=-1.0, high=1.0, shape=(4 + len(PLAYER_STATUSES),), dtype=np.float32),
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
            "action_mask": spaces.MultiBinary(MAX_ACTIONS),
        }
    )


def encode(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Encode one engine observation into a dict of numpy arrays."""
    return {
        "player": _encode_player(obs["player"]),
        "enemy": _encode_enemy(obs.get("enemy"), obs["player"]["maxHp"]),
        "enemy_present": np.array([1.0 if obs.get("enemy") else 0.0], dtype=np.float32),
        "phase": _encode_phase(obs["phase"]),
        "run": _encode_run(obs),
        "hand": _encode_hand(obs.get("hand") or []),
        "hand_mask": _encode_hand_mask(obs.get("hand") or []),
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
    if phase in PHASES:
        out[PHASES.index(phase)] = 1.0
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
        if card_type in CARD_TYPES:
            out[i, 2 + CARD_TYPES.index(card_type)] = 1.0
    return out


def _encode_hand_mask(hand: list[dict[str, Any]]) -> np.ndarray:
    mask = np.zeros(MAX_HAND_SIZE, dtype=np.int8)
    mask[: min(len(hand), MAX_HAND_SIZE)] = 1
    return mask


def _encode_action_mask(legal_actions: list[dict[str, Any]]) -> np.ndarray:
    mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
    mask[: min(len(legal_actions), MAX_ACTIONS)] = 1
    return mask


def _status_magnitude(status: dict[str, Any], key: str) -> float:
    entry = status.get(key)
    if entry is None:
        return 0.0
    if isinstance(entry, dict):
        amount = entry.get("amount", entry.get("value", entry.get("duration", 0)))
        return min(1.0, float(amount) / 5.0)
    if isinstance(entry, (int, float, bool)):
        return min(1.0, float(entry) / 5.0)
    return 0.0


def _normalize_cost(cost: Any) -> float:  # noqa: ANN401 — engine uses mixed types
    if cost == "X" or cost == -1:
        return 1.0
    try:
        return min(1.0, max(0.0, float(cost) / 3.0))
    except (TypeError, ValueError):
        return 0.0
