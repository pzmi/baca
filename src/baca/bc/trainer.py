"""Hand-rolled BC trainer over the frozen iter-2 ``MaskableBacaPolicy``.

This module composes Step 4 of the iter-3 plan: build a ``MaskableBacaPolicy``
without booting an RPC/engine subprocess, run a masked cross-entropy loop over
a ``BcDataset``, and save the trained policy as an SB3 checkpoint that
``MaskablePPO.load(..., env=env)`` can consume unchanged downstream.

SB3 implementation notes:

- ``policy.extract_features(obs)`` returns a single ``Tensor`` because
  ``share_features_extractor`` is ``True`` by default. Under a split extractor
  it would return a 2-tuple; we cover only the shared case (iter-2 convention).
- ``MaskablePPO.save`` serializes obs/action spaces + ``policy_kwargs`` +
  model attributes. To produce a zip consumable by ``MaskablePPO.load`` we
  construct a transient ``MaskablePPO`` around a no-op ``DummyVecEnv``, swap
  in our trained ``state_dict``, then call ``model.save``.
- Value-head freezing operates on ``policy.value_net`` plus any
  ``mlp_extractor.value_net`` sub-parameters. Both are ``nn.Module``s on the
  shared-extractor default.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from torch import nn

from baca.bc.dataset import BcDataset
from baca.encoder import MAX_ACTIONS, observation_space
from baca.policy import MaskableBacaPolicy


@dataclass(frozen=True)
class EpochMetrics:
    """Per-epoch BC training metrics emitted by :func:`train_bc`."""

    epoch: int
    train_loss: float
    holdout_accuracy: float
    holdout_mask_compliance: float


class _DummyEnv(gym.Env[dict[str, np.ndarray], int]):
    """No-op env matching BACA's obs/action spaces so SB3 can init the policy."""

    metadata = {"render_modes": []}  # noqa: RUF012

    def __init__(self, obs_space: spaces.Dict, action_space: spaces.Space[Any]) -> None:
        super().__init__()
        self.observation_space = obs_space
        self.action_space = action_space

    def _zero_obs(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for key, sub in self.observation_space.spaces.items():  # type: ignore[attr-defined]
            shape = tuple(int(x) for x in (sub.shape or ()))
            out[key] = np.zeros(shape, dtype=sub.dtype)
        return out

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        _ = seed, options
        return self._zero_obs(), {}

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        _ = action
        return self._zero_obs(), 0.0, True, False, {}

    def action_masks(self) -> np.ndarray:
        return np.ones(MAX_ACTIONS, dtype=bool)


def _make_dummy_vec_env(obs_space: spaces.Dict, action_space: spaces.Space[Any]) -> VecEnv:
    return DummyVecEnv([lambda: _DummyEnv(obs_space, action_space)])


def _transient_ppo(
    obs_space: spaces.Dict,
    action_space: spaces.Space[Any],
    *,
    policy_kwargs: dict[str, Any] | None,
    vec_env: VecEnv | None = None,
) -> tuple[MaskablePPO, VecEnv]:
    """Build a throw-away ``MaskablePPO`` so SB3 wires up policy + extractor."""
    env = vec_env if vec_env is not None else _make_dummy_vec_env(obs_space, action_space)
    model = MaskablePPO(
        MaskableBacaPolicy,
        env,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        verbose=0,
        policy_kwargs=dict(policy_kwargs or {}),
    )
    return model, env


def build_bc_policy(
    observation_space_: spaces.Dict | None = None,
    action_space: spaces.Space[Any] | None = None,
    *,
    policy_kwargs: dict[str, Any] | None = None,
) -> MaskableBacaPolicy:
    """Instantiate a fresh ``MaskableBacaPolicy`` without spawning the engine.

    Defaults to BACA's canonical observation/action spaces so callers that only
    want to train from ``BcDataset`` don't have to plumb spaces themselves.
    """
    obs_space = observation_space_ if observation_space_ is not None else observation_space()
    act_space = action_space if action_space is not None else spaces.Discrete(MAX_ACTIONS)
    model, _ = _transient_ppo(obs_space, act_space, policy_kwargs=policy_kwargs)
    policy = model.policy
    if not isinstance(policy, MaskableBacaPolicy):
        raise TypeError(f"expected MaskableBacaPolicy, got {type(policy).__name__}")
    return policy


def freeze_value_head(policy: MaskableBacaPolicy) -> int:
    """Freeze every parameter that contributes to value prediction.

    Touches ``policy.value_net`` and, if the shared-extractor MLP exposes a
    ``value_net`` branch, ``policy.mlp_extractor.value_net``. Returns the
    number of parameters actually frozen (for callers that want to sanity-check
    their policy against SB3's layer-naming drift).
    """
    targets: list[nn.Module] = []
    value_net = getattr(policy, "value_net", None)
    if isinstance(value_net, nn.Module):
        targets.append(value_net)
    mlp_extractor = getattr(policy, "mlp_extractor", None)
    if mlp_extractor is not None:
        mlp_value = getattr(mlp_extractor, "value_net", None)
        if isinstance(mlp_value, nn.Module):
            targets.append(mlp_value)

    frozen = 0
    for module in targets:
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad_(False)
            frozen += 1
    if frozen == 0:
        raise RuntimeError(
            "freeze_value_head found no value-head parameters to freeze; "
            "SB3 layer naming for MaskableMultiInputActorCriticPolicy may have drifted."
        )
    return frozen


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _select_holdout(dataset: BcDataset, holdout_frac: float, seed: int) -> set[Path]:
    if holdout_frac <= 0.0:
        return set()
    game_paths = [path for path, _ in dataset._index]
    if not game_paths:
        return set()
    k = max(1, int(len(game_paths) * holdout_frac))
    k = min(k, max(0, len(game_paths) - 1))
    if k == 0:
        return set()
    rng = random.Random(seed)  # noqa: S311 — split selection, not crypto
    return set(rng.sample(game_paths, k=k))


class _SplitDataset:
    """Lightweight view over ``BcDataset`` that restricts iteration to some games.

    Reuses the parent's ``iter_batches`` tensor materialization by temporarily
    swapping its private ``_index``. Safe because batches materialize per call
    and don't outlive ``iter_batches``'s generator.
    """

    def __init__(self, parent: BcDataset, allowed: set[Path]) -> None:
        self._parent = parent
        self._allowed_index = [entry for entry in parent._index if entry[0] in allowed]
        self._total = sum(n for _, n in self._allowed_index)

    def __len__(self) -> int:
        return self._total

    def iter_batches(
        self, batch_size: int, *, shuffle: bool = True, seed: int = 42
    ) -> Iterator[dict[str, torch.Tensor]]:
        parent = self._parent
        saved = parent._index
        parent._index = self._allowed_index
        parent._total_samples = self._total
        try:
            yield from parent.iter_batches(batch_size, shuffle=shuffle, seed=seed)
        finally:
            parent._index = saved
            parent._total_samples = sum(n for _, n in saved)


def _split_dataset(
    dataset: BcDataset, holdout_frac: float, seed: int
) -> tuple[_SplitDataset, _SplitDataset]:
    all_paths = {path for path, _ in dataset._index}
    holdout_paths = _select_holdout(dataset, holdout_frac, seed)
    train_paths = all_paths - holdout_paths
    return _SplitDataset(dataset, train_paths), _SplitDataset(dataset, holdout_paths)


def _move_batch_to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _split_obs_and_extras(
    batch: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Separate observation keys, action mask and action-index target."""
    obs = {k: v for k, v in batch.items() if k not in {"action_indices"}}
    action_mask = obs["action_mask"].bool()
    action_idx = batch["action_indices"].long()
    return obs, action_mask, action_idx


def _forward_logits(policy: MaskableBacaPolicy, obs: dict[str, torch.Tensor]) -> torch.Tensor:
    features = policy.extract_features(obs)
    # Split-extractor case returns (pi_features, vf_features); shared returns one tensor.
    pi_features = features[0] if isinstance(features, tuple) else features
    latent_pi = policy.mlp_extractor.forward_actor(pi_features)
    return policy.action_net(latent_pi)  # type: ignore[no-any-return]


def _masked_cross_entropy(
    logits: torch.Tensor, action_mask: torch.Tensor, action_idx: torch.Tensor
) -> torch.Tensor:
    neg_inf = torch.finfo(logits.dtype).min
    masked = logits.masked_fill(~action_mask, neg_inf)
    log_probs = torch.log_softmax(masked, dim=-1)
    batch_range = torch.arange(log_probs.shape[0], device=log_probs.device)
    return -log_probs[batch_range, action_idx].mean()


def _evaluate_holdout(
    policy: MaskableBacaPolicy,
    holdout: _SplitDataset,
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[float, float]:
    """Return ``(accuracy, mask_compliance)`` over the held-out games."""
    if len(holdout) == 0:
        return 0.0, 1.0
    policy.eval()
    total = 0
    correct = 0
    compliant = 0
    with torch.no_grad():
        for batch in holdout.iter_batches(batch_size, shuffle=False, seed=seed):
            batch = _move_batch_to_device(batch, device)
            obs, action_mask, action_idx = _split_obs_and_extras(batch)
            logits = _forward_logits(policy, obs)
            neg_inf = torch.finfo(logits.dtype).min
            masked = logits.masked_fill(~action_mask, neg_inf)
            pred = masked.argmax(dim=-1)
            batch_range = torch.arange(pred.shape[0], device=pred.device)
            correct += int((pred == action_idx).sum().item())
            compliant += int(action_mask[batch_range, pred].sum().item())
            total += int(pred.shape[0])
    if total == 0:
        return 0.0, 1.0
    return correct / total, compliant / total


def train_bc(
    policy: MaskableBacaPolicy,
    dataset: BcDataset,
    *,
    n_epochs: int,
    batch_size: int,
    lr: float,
    log_every: int = 50,
    holdout_frac: float = 0.05,
    seed: int = 42,
    device: torch.device | str | None = None,
) -> list[EpochMetrics]:
    """Run masked cross-entropy BC over ``dataset`` and return per-epoch metrics."""
    if n_epochs <= 0:
        raise ValueError(f"n_epochs must be positive; got {n_epochs}")
    _seed_everything(seed)
    resolved_device = torch.device(device) if device is not None else torch.device("cpu")
    policy.to(resolved_device)

    train_split, holdout_split = _split_dataset(dataset, holdout_frac, seed)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters — did freeze_value_head freeze everything?")
    optimizer = torch.optim.Adam(trainable, lr=lr)

    metrics: list[EpochMetrics] = []
    for epoch in range(n_epochs):
        policy.train()
        running_loss = 0.0
        running_batches = 0
        for step, batch in enumerate(
            train_split.iter_batches(batch_size, shuffle=True, seed=seed + epoch * 7919)
        ):
            batch = _move_batch_to_device(batch, resolved_device)
            obs, action_mask, action_idx = _split_obs_and_extras(batch)
            logits = _forward_logits(policy, obs)
            loss = _masked_cross_entropy(logits, action_mask, action_idx)
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            running_loss += float(loss.item())
            running_batches += 1
            if log_every > 0 and (step + 1) % log_every == 0:
                print(  # noqa: T201 — CLI progress output
                    f"[bc] epoch={epoch + 1}/{n_epochs} step={step + 1} "
                    f"loss={running_loss / running_batches:.4f}",
                    flush=True,
                )

        train_loss = running_loss / max(1, running_batches)
        holdout_acc, holdout_mask = _evaluate_holdout(
            policy, holdout_split, batch_size=batch_size, device=resolved_device, seed=seed
        )
        metrics.append(
            EpochMetrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                holdout_accuracy=holdout_acc,
                holdout_mask_compliance=holdout_mask,
            )
        )
        print(  # noqa: T201 — CLI progress output
            f"[bc] epoch={epoch + 1}/{n_epochs} train_loss={train_loss:.4f} "
            f"holdout_acc={holdout_acc:.3f} holdout_mask_compliance={holdout_mask:.3f}",
            flush=True,
        )

    policy.eval()
    return metrics


def save_bc_checkpoint(
    policy: MaskableBacaPolicy,
    observation_space_: spaces.Dict,
    action_space: spaces.Space[Any],
    out_path: Path,
    *,
    policy_kwargs: dict[str, Any] | None = None,
) -> None:
    """Save ``policy`` as a zip that ``MaskablePPO.load(path, env=...)`` accepts.

    Builds a transient ``MaskablePPO`` with matching ``policy_kwargs``, copies
    the trained ``state_dict`` into its policy, then calls ``model.save``.
    After writing, round-trips the zip through ``MaskablePPO.load`` + one
    ``predict`` call to fail loudly here instead of in Step 8's PPO wiring.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    transient_env = _make_dummy_vec_env(observation_space_, action_space)
    try:
        model, _ = _transient_ppo(
            observation_space_,
            action_space,
            policy_kwargs=policy_kwargs,
            vec_env=transient_env,
        )
        policy_cpu_state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
        model.policy.load_state_dict(policy_cpu_state)
        model.policy.set_training_mode(False)
        model.save(str(out_path))
    finally:
        transient_env.close()

    # Load roundtrip — SB3 has historically been brittle across torch versions
    # so prove locally that the checkpoint survives a full (load → predict) cycle.
    verify_env = _make_dummy_vec_env(observation_space_, action_space)
    try:
        try:
            loaded = MaskablePPO.load(str(out_path), env=verify_env)
        except Exception as exc:  # pragma: no cover — defensive
            raise RuntimeError(
                f"save_bc_checkpoint: MaskablePPO.load roundtrip failed for {out_path}"
            ) from exc
        obs_sample: dict[str, np.ndarray] = {}
        for key, sub in observation_space_.spaces.items():
            shape = tuple(int(x) for x in (sub.shape or ()))
            obs_sample[key] = np.zeros((1, *shape), dtype=sub.dtype)
        mask_sample = np.ones((1, MAX_ACTIONS), dtype=bool)
        loaded.predict(obs_sample, action_masks=mask_sample, deterministic=True)
    finally:
        verify_env.close()
