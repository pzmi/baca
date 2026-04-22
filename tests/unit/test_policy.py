"""Unit tests for :mod:`baca.policy` — the attention-pool features extractor."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from baca.encoder import CARD_VOCAB_SIZE, observation_space
from baca.policy import BacaFeaturesExtractor, _AttentionPool


def _zeros_obs(batch: int = 1) -> dict[str, torch.Tensor]:
    space = observation_space()
    out: dict[str, torch.Tensor] = {}
    for key, sub in space.spaces.items():
        sub_shape = sub.shape
        assert sub_shape is not None, f"{key} must have a concrete shape"
        shape = (batch, *tuple(sub_shape))
        if np.issubdtype(sub.dtype, np.integer):
            out[key] = torch.zeros(shape, dtype=torch.long)
        else:
            out[key] = torch.zeros(shape, dtype=torch.float32)
    return out


def test_shouldReturnFeaturesShapedAsFeaturesDimWhenForwardSingleObs() -> None:
    # given
    torch.manual_seed(0)
    extractor = BacaFeaturesExtractor(observation_space(), features_dim=64)
    obs = _zeros_obs(batch=1)

    # when
    out = extractor(obs)

    # then
    assert out.shape == (1, 64)


def test_shouldReturnFeaturesShapedAsFeaturesDimWhenForwardBatchedObs() -> None:
    # given
    torch.manual_seed(0)
    extractor = BacaFeaturesExtractor(observation_space(), features_dim=64)
    obs = _zeros_obs(batch=4)

    # when
    out = extractor(obs)

    # then
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()


def test_shouldProduceZeroAttentionOverPaddedSlotsWhenMaskBitsZero() -> None:
    # given — two identical observations differing only at a padded hand slot
    torch.manual_seed(0)
    extractor = BacaFeaturesExtractor(observation_space(), features_dim=64)
    extractor.eval()
    obs_a = _zeros_obs(batch=1)
    obs_b = _zeros_obs(batch=1)
    # Only the first hand slot is active, rest are padding.
    obs_a["hand_mask"][0, 0] = 1
    obs_b["hand_mask"][0, 0] = 1
    obs_a["hand_ids"][0, 0] = 5
    obs_b["hand_ids"][0, 0] = 5
    # Flip a padded slot — the mask should zero it out of the pool.
    obs_a["hand_ids"][0, 3] = 42
    obs_b["hand_ids"][0, 3] = 7
    obs_a["hand"][0, 3, 0] = 0.9
    obs_b["hand"][0, 3, 0] = 0.1

    # when
    with torch.no_grad():
        out_a = extractor(obs_a)
        out_b = extractor(obs_b)

    # then — padded-slot content must not leak into the pooled output.
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_shouldProduceZeroPoolWhenAllSlotsMaskedOut() -> None:
    # given — reward block has no valid slots; softmax would NaN without the guard.
    torch.manual_seed(0)
    extractor = BacaFeaturesExtractor(observation_space(), features_dim=64)
    obs = _zeros_obs(batch=2)

    # when
    out = extractor(obs)

    # then
    assert torch.isfinite(out).all()


def test_shouldFailForwardWhenObservationMissingRequiredKey() -> None:
    # given
    torch.manual_seed(0)
    extractor = BacaFeaturesExtractor(observation_space(), features_dim=64)
    obs = _zeros_obs(batch=1)
    del obs["reward_mask"]

    # when / then
    with pytest.raises(KeyError):
        extractor(obs)


def test_shouldRejectObservationSpaceMissingKeysWhenConstructed() -> None:
    # given
    from gymnasium import spaces

    # Drop one of the required keys from the space.
    full = observation_space()
    incomplete = spaces.Dict({k: v for k, v in full.spaces.items() if k != "hand_ids"})

    # when / then
    with pytest.raises(ValueError, match="hand_ids"):
        BacaFeaturesExtractor(incomplete)


def _make_populated_obs(batch: int) -> dict[str, torch.Tensor]:
    """Obs with all hand/reward/shop slots valid and random card ids."""
    obs = _zeros_obs(batch=batch)
    obs["hand_mask"] = torch.ones_like(obs["hand_mask"], dtype=torch.float32)
    obs["reward_mask"] = torch.ones_like(obs["reward_mask"], dtype=torch.float32)
    obs["shop_mask"] = torch.ones_like(obs["shop_mask"], dtype=torch.float32)
    rng = torch.Generator().manual_seed(7)
    obs["hand_ids"] = torch.randint(1, CARD_VOCAB_SIZE, obs["hand_ids"].shape, generator=rng)
    obs["reward_ids"] = torch.randint(1, CARD_VOCAB_SIZE, obs["reward_ids"].shape, generator=rng)
    obs["shop_ids"] = torch.randint(1, CARD_VOCAB_SIZE, obs["shop_ids"].shape, generator=rng)
    return obs


def test_shouldKeepGradientsBoundedAtInitWhenForwardPopulatedObs() -> None:
    # given — realistic obs with multi-slot blocks so softmax produces non-trivial grads.
    torch.manual_seed(0)
    extractor = BacaFeaturesExtractor(observation_space())
    extractor.train()
    obs = _make_populated_obs(batch=4)

    # when
    features = extractor(obs)
    features.sum().backward()

    # then — no NaN / Inf anywhere, and attention queries + final MLP are alive.
    for name, param in extractor.named_parameters():
        if param.grad is None:
            continue
        assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"
        assert not torch.isinf(param.grad).any(), f"Inf grad in {name}"

    def _grad_mean_abs(tensor: torch.Tensor) -> float:
        grad = tensor.grad
        assert grad is not None
        return float(grad.abs().mean().item())

    assert _grad_mean_abs(extractor._hand_pool._query) > 0.0
    assert _grad_mean_abs(extractor._reward_pool._query) > 0.0
    assert _grad_mean_abs(extractor._shop_pool._query) > 0.0
    final_first_linear = extractor._final_mlp[0]
    assert isinstance(final_first_linear, torch.nn.Linear)
    assert _grad_mean_abs(final_first_linear.weight) > 0.0


def test_shouldProduceNegativeFeatureValuesWhenForwardAllowsSignedActivations() -> None:
    """Sentinel against re-adding a terminal ReLU in ``_final_mlp``.

    SB3's ``ortho_init=True`` on policy/value heads atop a non-negative feature
    vector halves the effective input dim the heads can exploit. Iter-2-norelu
    dropped the trailing ``nn.ReLU()`` so features span ``R^features_dim``; this
    test proves at least one sign-flipped activation slips through, which only
    happens if that ReLU is really gone.
    """
    # given
    torch.manual_seed(0)
    extractor = BacaFeaturesExtractor(observation_space())
    extractor.eval()
    obs = _make_populated_obs(batch=8)

    # when
    with torch.no_grad():
        features = extractor(obs)

    # then
    assert features.shape[1] == 64
    assert (features < 0.0).any(), (
        "features are all non-negative — a terminal ReLU likely crept back "
        "into _final_mlp, cutting the SB3 head's effective input dim in half."
    )


def test_shouldProduceNonUniformAttentionWeightsAtInit() -> None:
    """Healthy-init sentry.

    ``_AttentionPool`` must produce a clearly non-uniform weight distribution
    at initialization so the query parameter receives meaningful gradient from
    the first PPO update onwards. Iter-2 shipped a double-scale bug that drove
    logit std toward ``1/D`` and collapsed softmax to uniform; the fix drops
    the separate scale and inits ``_query`` with std=1 (no ``1/sqrt(D)``
    division).
    """
    # given
    torch.manual_seed(0)
    extractor = BacaFeaturesExtractor(observation_space())
    extractor.eval()
    obs = _make_populated_obs(batch=4)

    # when — monkey-patch the pool to capture weights without changing semantics.
    captured: dict[str, torch.Tensor] = {}
    original = _AttentionPool.forward

    def instrumented(
        self: _AttentionPool, features: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        logits = torch.einsum("bnd,d->bn", features, self._query)
        mask_bool = mask > 0.5
        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~mask_bool, neg_inf)
        any_valid = mask_bool.any(dim=-1, keepdim=True)
        safe_logits = torch.where(any_valid, logits, torch.zeros_like(logits))
        weights = torch.softmax(safe_logits, dim=-1)
        weights = weights * any_valid.to(weights.dtype)
        captured[str(id(self))] = weights.detach()
        return torch.einsum("bn,bnd->bd", weights, features)

    _AttentionPool.forward = instrumented  # type: ignore[method-assign]
    try:
        with torch.no_grad():
            extractor(obs)
    finally:
        _AttentionPool.forward = original  # type: ignore[method-assign]

    # then — hand block weights peak clearly above uniform and spread visibly.
    hand_weights = captured[str(id(extractor._hand_pool))]
    assert hand_weights.shape == (4, 10)
    uniform = 1.0 / hand_weights.shape[-1]
    assert hand_weights.max().item() > 1.2 * uniform, (
        "attention collapsed toward uniform at init — init recipe regressed"
    )
    assert hand_weights.std(dim=-1).mean().item() > 0.05, (
        "attention weights too flat at init — init recipe regressed"
    )
