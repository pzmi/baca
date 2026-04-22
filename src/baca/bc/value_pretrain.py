"""Value-head pretraining on discounted returns from the BC dataset.

Step 7 of the iter-3 plan. After BC converges, we freeze the features extractor
plus policy head and fit ``policy.value_net`` (and ``policy.mlp_extractor.value_net``)
to the MC-return targets computed from the HeuristicBot rollouts. The resulting
checkpoint drops into Step 8's PPO warm-start so the first rollouts' advantages
are not centered on random value estimates.

Reward contract (must match :meth:`baca.env.UsiecCepraEnv._compute_reward` with
``reward_shape="floor"``):

- Intermediate steps: 0.
- Terminal step (only when ``outcome == "player_win"`` or ``"player_lose"``):
  ``0.1 * min(1, floor_reached/15) + 0.9 * (outcome == "player_win")``.
- Truncated episodes: 0 at terminal too (env returns ``is_win=0`` + ``done=False``
  when truncated; our manifest labels those ``"truncated"`` via env.step's info,
  so anything that is not ``"player_win"`` / ``"player_lose"`` gets 0 reward).
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from torch.nn import functional as torch_fn

from baca.bc.dataset import BcDataset
from baca.bc.generate_cli import MANIFEST_NAME
from baca.bc.trainer import save_bc_checkpoint
from baca.policy import MaskableBacaPolicy

_MASK_KEYS = frozenset({"hand_mask", "reward_mask", "shop_mask", "action_mask"})
_INT_KEYS = frozenset({"hand_ids", "reward_ids", "shop_ids"})


@dataclass(frozen=True)
class EpochMetrics:
    """Per-epoch value-pretrain metrics."""

    epoch: int
    train_loss: float
    val_pred_mean: float
    val_pred_std: float
    returns_mean: float
    returns_std: float


def compute_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Return per-step MC returns ``G_t = sum_{k=0}^{T-t-1} gamma^k r_{t+k}``.

    ``rewards`` must be 1-D float. Empty input returns an empty float32 array.
    """
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be 1-D; got shape {rewards.shape}")
    n = int(rewards.shape[0])
    out = np.zeros(n, dtype=np.float32)
    if n == 0:
        return out
    running = 0.0
    for t in range(n - 1, -1, -1):
        running = float(rewards[t]) + gamma * running
        out[t] = running
    return out


def build_rewards_from_manifest(n_steps: int, outcome: str, floor_reached: int) -> np.ndarray:
    """Reconstruct the per-step reward vector matching ``env._compute_reward``.

    Zero on every non-terminal step. Terminal step gets
    ``0.1 * min(1, f/15) + 0.9 * is_win`` when the episode ended with a
    win/loss outcome; truncated or unknown outcomes get 0 terminal reward.
    """
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative; got {n_steps}")
    rewards = np.zeros(n_steps, dtype=np.float32)
    if n_steps == 0:
        return rewards
    # Only player_win / player_lose are "terminal" outcomes with env reward.
    # Anything else (truncated, unknown, empty) contributes 0.
    if outcome not in {"player_win", "player_lose"}:
        return rewards
    is_win = 1.0 if outcome == "player_win" else 0.0
    floor = max(0.0, float(floor_reached))
    floor_progress = min(1.0, floor / 15.0)
    terminal = 0.1 * floor_progress + 0.9 * is_win
    rewards[-1] = terminal
    return rewards


def freeze_policy_side(policy: MaskableBacaPolicy) -> int:
    """Freeze everything except the value-head parameters.

    Contract (plan §4 D4): freeze features extractor, policy branch of the shared
    MLP extractor, and the action head. Leaves ``policy.value_net`` and
    ``policy.mlp_extractor.value_net`` trainable so the value head fits MC targets
    under the representation BC converged to. Returns the number of frozen
    parameters.
    """
    targets: list[nn.Module] = []
    features_extractor = getattr(policy, "features_extractor", None)
    if isinstance(features_extractor, nn.Module):
        targets.append(features_extractor)
    mlp_extractor = getattr(policy, "mlp_extractor", None)
    if mlp_extractor is not None:
        policy_branch = getattr(mlp_extractor, "policy_net", None)
        if isinstance(policy_branch, nn.Module):
            targets.append(policy_branch)
    action_net = getattr(policy, "action_net", None)
    if isinstance(action_net, nn.Module):
        targets.append(action_net)

    frozen = 0
    for module in targets:
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad_(False)
            frozen += 1
    if frozen == 0:
        raise RuntimeError(
            "freeze_policy_side found no policy-side parameters; SB3 layer naming "
            "for MaskableMultiInputActorCriticPolicy may have drifted."
        )
    return frozen


def _load_manifest(root: Path) -> list[dict[str, Any]]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest missing: {manifest_path}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"manifest {manifest_path} is not a list")
    return raw


def _load_game_obs_and_meta(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return ``(encoded_obs, meta)`` for one NPZ file.

    ``meta`` exposes the scalars the manifest also tracks; we prefer the NPZ
    to avoid a manifest/NPZ drift surprise inside the loader.
    """
    with np.load(path, mmap_mode="r") as data:
        encoded_obs: dict[str, np.ndarray] = {}
        for key in data.files:
            if key.startswith("obs_"):
                encoded_obs[key[len("obs_") :]] = np.asarray(data[key])
        meta: dict[str, Any] = {
            "n_steps": int(np.asarray(data["n_steps"])),
            "outcome": str(np.asarray(data["outcome"])),
            "floor_reached": int(np.asarray(data["floor_reached"])),
        }
    return encoded_obs, meta


def _to_tensor(key: str, arr: np.ndarray) -> torch.Tensor:
    if key in _MASK_KEYS:
        return torch.as_tensor(arr.astype(np.bool_), dtype=torch.bool)
    if key in _INT_KEYS:
        return torch.as_tensor(arr.astype(np.int64), dtype=torch.long)
    return torch.as_tensor(arr.astype(np.float32), dtype=torch.float32)


def _value_batches(
    dataset: BcDataset,
    *,
    gamma: float,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[dict[str, torch.Tensor | dict[str, torch.Tensor]]]:
    """Stream batches of ``(obs_dict, returns)`` over all games.

    Games are processed whole (in original step order so returns are correct),
    then appended to a buffer that we flush in shuffled batches of
    ``batch_size``. Buffer capacity mirrors ``BcDataset.iter_batches`` so memory
    behavior is predictable.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size}")

    manifest = _load_manifest(dataset.root)
    game_order = list(range(len(manifest)))
    if shuffle:
        random.Random(seed).shuffle(game_order)  # noqa: S311 — not crypto

    buffer_capacity = batch_size * 4
    obs_buffer: dict[str, list[np.ndarray]] = {}
    returns_buffer: list[float] = []

    def flush(size: int) -> Iterator[dict[str, torch.Tensor | dict[str, torch.Tensor]]]:
        nonlocal obs_buffer, returns_buffer
        while len(returns_buffer) >= size:
            if shuffle:
                gen = torch.Generator()
                gen.manual_seed(seed + len(returns_buffer))
                perm = torch.randperm(len(returns_buffer), generator=gen).tolist()
            else:
                perm = list(range(len(returns_buffer)))
            take = perm[:size]
            keep = perm[size:]
            batch = _materialize(obs_buffer, returns_buffer, take)
            obs_buffer = {k: [v[i] for i in keep] for k, v in obs_buffer.items()}
            returns_buffer = [returns_buffer[i] for i in keep]
            yield batch

    for idx in game_order:
        entry = manifest[idx]
        filename = entry.get("filename")
        if not isinstance(filename, str):
            continue
        n_steps_manifest = int(entry.get("n_steps", 0))
        if n_steps_manifest <= 0:
            continue
        file_path = dataset.root / filename
        encoded_obs, meta = _load_game_obs_and_meta(file_path)
        n_steps = int(meta["n_steps"])
        if n_steps <= 0:
            continue
        rewards = build_rewards_from_manifest(
            n_steps=n_steps,
            outcome=str(meta["outcome"]),
            floor_reached=int(meta["floor_reached"]),
        )
        returns = compute_returns(rewards, gamma)
        for step in range(n_steps):
            for key, arr in encoded_obs.items():
                obs_buffer.setdefault(key, []).append(np.asarray(arr[step]))
            returns_buffer.append(float(returns[step]))
        if len(returns_buffer) >= buffer_capacity:
            yield from flush(batch_size)

    while len(returns_buffer) >= batch_size:
        yield from flush(batch_size)
    if returns_buffer:
        yield _materialize(obs_buffer, returns_buffer, list(range(len(returns_buffer))))


def _materialize(
    obs_buffer: dict[str, list[np.ndarray]],
    returns_buffer: list[float],
    indices: list[int],
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    obs_dict: dict[str, torch.Tensor] = {}
    for key, rows in obs_buffer.items():
        stacked = np.stack([rows[i] for i in indices], axis=0)
        obs_dict[key] = _to_tensor(key, stacked)
    returns_tensor = torch.as_tensor([returns_buffer[i] for i in indices], dtype=torch.float32)
    return {"obs_dict": obs_dict, "returns": returns_tensor}


def _move_obs_to_device(
    obs: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in obs.items()}


def _forward_values(policy: MaskableBacaPolicy, obs: dict[str, torch.Tensor]) -> torch.Tensor:
    features = policy.extract_features(obs)
    if isinstance(features, tuple):
        # Split-extractor case: (pi_features, vf_features). Iter-2 uses the
        # shared extractor, but cover both for SB3 drift safety.
        vf_features = features[1]
        latent_vf = policy.mlp_extractor.forward_critic(vf_features)
    else:
        _, latent_vf = policy.mlp_extractor(features)
    values: torch.Tensor = policy.value_net(latent_vf).squeeze(-1)
    return values


def train_value_head(
    policy: MaskableBacaPolicy,
    dataset: BcDataset,
    *,
    n_epochs: int = 2,
    batch_size: int = 256,
    lr: float = 1e-3,
    gamma: float = 0.99,
    log_every: int = 50,
    seed: int = 42,
    device: torch.device | str | None = None,
) -> list[EpochMetrics]:
    """Fit ``policy.value_net`` to MC returns computed from ``dataset``.

    Expects :func:`freeze_policy_side` to have been called on ``policy``. The
    optimizer collects only ``requires_grad`` params so the contract is enforced
    downstream too.
    """
    if n_epochs <= 0:
        raise ValueError(f"n_epochs must be positive; got {n_epochs}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    resolved_device = torch.device(device) if device is not None else torch.device("cpu")
    policy.to(resolved_device)

    trainable = [p for p in policy.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters — did freeze_policy_side freeze everything?")
    optimizer = torch.optim.Adam(trainable, lr=lr)

    metrics: list[EpochMetrics] = []
    for epoch in range(n_epochs):
        policy.train()
        running_loss = 0.0
        running_batches = 0
        pred_sum = 0.0
        pred_sq_sum = 0.0
        returns_sum = 0.0
        returns_sq_sum = 0.0
        total_samples = 0
        for step, batch in enumerate(
            _value_batches(
                dataset,
                gamma=gamma,
                batch_size=batch_size,
                shuffle=True,
                seed=seed + epoch * 7919,
            )
        ):
            obs = _move_obs_to_device(batch["obs_dict"], resolved_device)  # type: ignore[arg-type]
            returns = batch["returns"].to(resolved_device)  # type: ignore[union-attr]
            values = _forward_values(policy, obs)
            loss = torch_fn.mse_loss(values, returns)
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

            running_loss += float(loss.item())
            running_batches += 1
            with torch.no_grad():
                batch_n = int(values.shape[0])
                total_samples += batch_n
                pred_sum += float(values.sum().item())
                pred_sq_sum += float((values * values).sum().item())
                returns_sum += float(returns.sum().item())
                returns_sq_sum += float((returns * returns).sum().item())

            if log_every > 0 and (step + 1) % log_every == 0:
                print(  # noqa: T201 — CLI progress output
                    f"[value-pretrain] epoch={epoch + 1}/{n_epochs} step={step + 1} "
                    f"loss={running_loss / running_batches:.5f}",
                    flush=True,
                )

        train_loss = running_loss / max(1, running_batches)
        if total_samples > 0:
            pred_mean = pred_sum / total_samples
            pred_var = max(0.0, pred_sq_sum / total_samples - pred_mean * pred_mean)
            returns_mean = returns_sum / total_samples
            returns_var = max(0.0, returns_sq_sum / total_samples - returns_mean * returns_mean)
        else:
            pred_mean = 0.0
            pred_var = 0.0
            returns_mean = 0.0
            returns_var = 0.0
        metrics.append(
            EpochMetrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_pred_mean=pred_mean,
                val_pred_std=float(np.sqrt(pred_var)),
                returns_mean=returns_mean,
                returns_std=float(np.sqrt(returns_var)),
            )
        )
        print(  # noqa: T201 — CLI progress output
            f"[value-pretrain] epoch={epoch + 1}/{n_epochs} train_loss={train_loss:.5f} "
            f"pred_mean={pred_mean:.4f} pred_std={float(np.sqrt(pred_var)):.4f} "
            f"returns_mean={returns_mean:.4f} returns_std={float(np.sqrt(returns_var)):.4f}",
            flush=True,
        )

    policy.eval()
    return metrics


def save_value_checkpoint(
    policy: MaskableBacaPolicy,
    observation_space_: spaces.Dict,
    action_space: spaces.Space[Any],
    out_path: Path,
    *,
    policy_kwargs: dict[str, Any] | None = None,
) -> None:
    """Save the BC+value-pretrained policy as an SB3-loadable zip.

    Delegates to :func:`baca.bc.trainer.save_bc_checkpoint`; the zip format is
    identical. Step 8's PPO warm-start loads either checkpoint uniformly.
    """
    save_bc_checkpoint(
        policy, observation_space_, action_space, out_path, policy_kwargs=policy_kwargs
    )
