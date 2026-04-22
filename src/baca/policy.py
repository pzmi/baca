"""Custom MaskablePPO policy with card-id embedding + attention pool.

Consumes the :class:`gymnasium.spaces.Dict` produced by :func:`baca.encoder.encode`
and projects it to a fixed-width feature vector. The key idea is that
hand/shop/reward are variable-length card lists; we embed each card id, project
per-slot features, then pool via single-head scaled-dot-product attention with a
learned block query — so padded slots contribute zero regardless of what id
they carry. Scalar blocks (player/enemy/run/phase/action_mask) are concatenated
and MLP-projected.

Kept as a standalone module so ``MaskablePPO.load(...)`` in ``eval_cli`` can
import the extractor without pulling training deps.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from baca.encoder import CARD_VOCAB_SIZE

# Keys the extractor reads. Missing keys raise at forward time rather than being
# silently zero-filled (defensive: encoder drift should surface loudly).
_REQUIRED_KEYS: tuple[str, ...] = (
    "player",
    "enemy",
    "enemy_present",
    "phase",
    "run",
    "hand",
    "hand_mask",
    "hand_ids",
    "reward_ids",
    "reward_mask",
    "reward_costs",
    "shop_ids",
    "shop_mask",
    "shop_costs",
    "action_mask",
)


class _AttentionPool(nn.Module):
    """Single-head scaled dot-product attention with a learned block query.

    Given a batch ``(B, N, D)`` of slot embeddings plus a mask ``(B, N)`` with
    1.0 for real slots and 0.0 for padding, returns a pooled vector of shape
    ``(B, D)``. Padded slots receive ``-inf`` logits so they contribute zero
    weight regardless of their embedded values.
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        # Init with std=1 (no `1/sqrt(D)` division) and no separate scale on the
        # dot product. Iter-2's double-scale shrank logit std to ≈ 1/D, making
        # the softmax near-uniform at init and starving the query of gradient.
        self._query = nn.Parameter(torch.randn(feature_dim))

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # features: (B, N, D), mask: (B, N) with 1.0 for valid slots.
        logits = torch.einsum("bnd,d->bn", features, self._query)
        # Convert mask (0/1) to additive bias (-inf where masked).
        mask_bool = mask > 0.5
        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~mask_bool, neg_inf)
        # If every slot is masked, softmax would produce NaN. Fall back to zero
        # weights; multiplying by features still yields zeros.
        any_valid = mask_bool.any(dim=-1, keepdim=True)
        safe_logits = torch.where(any_valid, logits, torch.zeros_like(logits))
        weights = torch.softmax(safe_logits, dim=-1)
        weights = weights * any_valid.to(weights.dtype)
        return torch.einsum("bn,bnd->bd", weights, features)


class BacaFeaturesExtractor(BaseFeaturesExtractor):
    """Card-embedding + attention-pool feature extractor for Usiec Cepra.

    Output width is ``features_dim`` (default 64). The pipeline:

    1. Shared ``nn.Embedding(CARD_VOCAB_SIZE, card_embed_dim)`` over hand / shop
       / reward id blocks.
    2. Per-block MLP projects ``[embed(id), *scalar_features]`` to ``block_dim``.
    3. Attention pool per block using the block's mask to zero out padding.
    4. Scalar features (player / enemy / run / phase / action_mask) concatenate
       and project through an MLP to ``block_dim``.
    5. Concatenate the four ``block_dim`` vectors, project to ``features_dim``.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        card_embed_dim: int = 32,
        block_dim: int = 64,
        features_dim: int = 64,
    ) -> None:
        super().__init__(observation_space, features_dim=features_dim)
        missing = [k for k in _REQUIRED_KEYS if k not in observation_space.spaces]
        if missing:
            raise ValueError(f"BacaFeaturesExtractor: observation_space missing keys {missing}")

        self._card_embed_dim = card_embed_dim
        self._block_dim = block_dim
        self._embedding = nn.Embedding(CARD_VOCAB_SIZE, card_embed_dim)

        # Per-slot feature widths for each list block:
        # hand slot: embed + hand feature row (7 dims: cost + isPlayable + one-hot types) + mask bit
        hand_shape = _require_shape(observation_space, "hand")
        hand_feat_dim = int(np.prod(hand_shape[1:]))
        hand_slot_in = card_embed_dim + hand_feat_dim + 1
        self._hand_slot_mlp = _slot_mlp(hand_slot_in, block_dim)
        self._hand_pool = _AttentionPool(block_dim)

        # shop slot: embed + cost_scalar + mask_bit
        shop_slot_in = card_embed_dim + 1 + 1
        self._shop_slot_mlp = _slot_mlp(shop_slot_in, block_dim)
        self._shop_pool = _AttentionPool(block_dim)

        # reward slot: embed + cost_scalar + mask_bit (same shape as shop)
        reward_slot_in = card_embed_dim + 1 + 1
        self._reward_slot_mlp = _slot_mlp(reward_slot_in, block_dim)
        self._reward_pool = _AttentionPool(block_dim)

        scalar_in = sum(
            int(np.prod(_require_shape(observation_space, key)))
            for key in ("player", "enemy", "enemy_present", "phase", "run", "action_mask")
        )
        self._scalar_mlp = nn.Sequential(
            nn.Linear(scalar_in, block_dim),
            nn.ReLU(),
        )

        # No terminal ReLU: SB3 applies ortho_init to the policy/value heads
        # atop these features, and a non-negative feature vector halves the
        # effective input dimensionality those heads can exploit.
        self._final_mlp = nn.Sequential(
            nn.Linear(4 * block_dim, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        missing = [k for k in _REQUIRED_KEYS if k not in observations]
        if missing:
            raise KeyError(f"Observation dict missing required keys: {missing}")

        hand_ids = observations["hand_ids"].long()
        reward_ids = observations["reward_ids"].long()
        shop_ids = observations["shop_ids"].long()

        hand_embed = self._embedding(hand_ids)  # (B, H, E)
        reward_embed = self._embedding(reward_ids)  # (B, R, E)
        shop_embed = self._embedding(shop_ids)  # (B, S, E)

        hand_feats = observations["hand"].float()  # (B, H, F)
        hand_mask = observations["hand_mask"].float()  # (B, H)
        hand_slot_inputs = torch.cat([hand_embed, hand_feats, hand_mask.unsqueeze(-1)], dim=-1)
        hand_slots = self._hand_slot_mlp(hand_slot_inputs)
        hand_vec = self._hand_pool(hand_slots, hand_mask)

        reward_mask = observations["reward_mask"].float()
        reward_costs = observations["reward_costs"].float().unsqueeze(-1)
        reward_slot_inputs = torch.cat(
            [reward_embed, reward_costs, reward_mask.unsqueeze(-1)], dim=-1
        )
        reward_slots = self._reward_slot_mlp(reward_slot_inputs)
        reward_vec = self._reward_pool(reward_slots, reward_mask)

        shop_mask = observations["shop_mask"].float()
        shop_costs = observations["shop_costs"].float().unsqueeze(-1)
        shop_slot_inputs = torch.cat([shop_embed, shop_costs, shop_mask.unsqueeze(-1)], dim=-1)
        shop_slots = self._shop_slot_mlp(shop_slot_inputs)
        shop_vec = self._shop_pool(shop_slots, shop_mask)

        scalars = torch.cat(
            [
                observations["player"].float(),
                observations["enemy"].float(),
                observations["enemy_present"].float(),
                observations["phase"].float(),
                observations["run"].float(),
                observations["action_mask"].float(),
            ],
            dim=-1,
        )
        scalar_vec = self._scalar_mlp(scalars)

        combined = torch.cat([hand_vec, reward_vec, shop_vec, scalar_vec], dim=-1)
        out: torch.Tensor = self._final_mlp(combined)
        return out


def _slot_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ReLU(),
    )


def _require_shape(space: spaces.Dict, key: str) -> tuple[int, ...]:
    shape = space.spaces[key].shape
    if shape is None:
        raise ValueError(f"observation_space[{key!r}] has unknown shape")
    return tuple(shape)


class MaskableBacaPolicy(MaskableMultiInputActorCriticPolicy):
    """MaskablePPO policy that uses :class:`BacaFeaturesExtractor` by default."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space[Any],
        lr_schedule: Callable[[float], float],
        *args: Any,
        card_embed_dim: int = 32,
        block_dim: int = 64,
        features_dim: int = 64,
        **kwargs: Any,
    ) -> None:
        # SB3 re-instantiates policies from saved kwargs on load; strip keys
        # we manage ourselves so the super() call never receives them twice.
        kwargs.pop("features_extractor_class", None)
        extractor_kwargs: dict[str, Any] = kwargs.pop("features_extractor_kwargs", None) or {}
        extractor_kwargs.setdefault("card_embed_dim", card_embed_dim)
        extractor_kwargs.setdefault("block_dim", block_dim)
        extractor_kwargs.setdefault("features_dim", features_dim)
        kwargs["features_extractor_class"] = BacaFeaturesExtractor
        kwargs["features_extractor_kwargs"] = extractor_kwargs
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
