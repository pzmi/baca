"""Streaming BC dataset reader.

``BcDataset`` indexes an NPZ-per-game directory produced by
``baca.bc.generate_cli`` and yields cross-game shuffled batches of torch
tensors suitable for the BC trainer. Samples never leave memory as a whole;
the iterator buffers ``batch_size * 4`` samples at a time so cross-game
shuffling works without a full-dataset load.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from baca.bc.generate_cli import MANIFEST_NAME, SNAPSHOT_NAME
from baca.encoder import CARD_IDS, observation_space

_MASK_KEYS = frozenset({"hand_mask", "reward_mask", "shop_mask", "action_mask"})
_INT_KEYS = frozenset({"hand_ids", "reward_ids", "shop_ids"})


def _validate_snapshot(root: Path) -> None:
    snapshot_path = root / SNAPSHOT_NAME
    if not snapshot_path.is_file():
        raise ValueError(
            f"BcDataset at {root} is missing {SNAPSHOT_NAME}; was it produced by generate_cli?"
        )
    recorded = snapshot_path.read_text(encoding="utf-8").strip()
    current = json.dumps(list(CARD_IDS))
    if recorded != current:
        raise ValueError(
            f"CARD_IDS mismatch for dataset at {root}: snapshot differs from current "
            f"baca.encoder.CARD_IDS. Re-generate the dataset or pin the engine."
        )


def _load_game_npz(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load one NPZ, unflattening ``obs_<key>`` back into a dict of arrays."""
    with np.load(path, mmap_mode="r") as data:
        encoded_obs: dict[str, np.ndarray] = {}
        for key in data.files:
            if key.startswith("obs_"):
                encoded_obs[key[len("obs_") :]] = np.asarray(data[key])
        action_indices = np.asarray(data["action_indices"], dtype=np.int64)
    return encoded_obs, action_indices


class BcDataset:
    """NPZ-per-game BC dataset with a streaming batch iterator."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root).resolve()
        if not self._root.is_dir():
            raise FileNotFoundError(f"BcDataset root not found: {self._root}")
        _validate_snapshot(self._root)

        manifest_path = self._root / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Dataset manifest missing: {manifest_path}")
        manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest_raw, list):
            raise ValueError(f"manifest {manifest_path} is not a list")

        self._index: list[tuple[Path, int]] = []
        for entry in manifest_raw:
            filename = entry.get("filename")
            n_steps = int(entry.get("n_steps", 0))
            if not isinstance(filename, str):
                raise ValueError(f"manifest entry missing filename: {entry!r}")
            if n_steps <= 0:
                continue
            self._index.append((self._root / filename, n_steps))
        self._total_samples = sum(n for _, n in self._index)

        self._observation_space = observation_space()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def game_count(self) -> int:
        return len(self._index)

    def __len__(self) -> int:
        return self._total_samples

    def iter_batches(
        self,
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 42,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Stream batches as dicts of torch tensors.

        Each batch tensor shape is ``(B, *inner_shape)`` matching the per-key
        shape in :func:`baca.encoder.observation_space`, plus ``action_indices``
        of shape ``(B,)`` dtype ``int64``.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {batch_size}")

        game_order = list(range(len(self._index)))
        if shuffle:
            random.Random(seed).shuffle(game_order)  # noqa: S311 — shuffle order, not crypto

        buffer_capacity = batch_size * 4
        obs_buffer: dict[str, list[np.ndarray]] = {}
        action_buffer: list[int] = []

        def flush(size: int) -> Iterator[dict[str, torch.Tensor]]:
            nonlocal obs_buffer, action_buffer
            while len(action_buffer) >= size:
                if shuffle:
                    gen = torch.Generator()
                    gen.manual_seed(seed + len(action_buffer))
                    perm = torch.randperm(len(action_buffer), generator=gen).tolist()
                else:
                    perm = list(range(len(action_buffer)))
                take = perm[:size]
                keep = perm[size:]
                batch = self._materialize_batch(obs_buffer, action_buffer, take)
                obs_buffer = {k: [v[i] for i in keep] for k, v in obs_buffer.items()}
                action_buffer = [action_buffer[i] for i in keep]
                yield batch

        for game_idx in game_order:
            file_path, _ = self._index[game_idx]
            encoded_obs, action_indices = _load_game_npz(file_path)
            for step in range(action_indices.shape[0]):
                for key, arr in encoded_obs.items():
                    obs_buffer.setdefault(key, []).append(np.asarray(arr[step]))
                action_buffer.append(int(action_indices[step]))
            if len(action_buffer) >= buffer_capacity:
                yield from flush(batch_size)

        while len(action_buffer) >= batch_size:
            yield from flush(batch_size)
        if action_buffer:
            yield self._materialize_batch(
                obs_buffer, action_buffer, list(range(len(action_buffer)))
            )

    def _materialize_batch(
        self,
        obs_buffer: dict[str, list[np.ndarray]],
        action_buffer: list[int],
        indices: list[int],
    ) -> dict[str, torch.Tensor]:
        batch: dict[str, torch.Tensor] = {}
        for key, rows in obs_buffer.items():
            stacked = np.stack([rows[i] for i in indices], axis=0)
            batch[key] = _to_tensor(key, stacked)
        batch["action_indices"] = torch.as_tensor(
            [action_buffer[i] for i in indices], dtype=torch.long
        )
        return batch


def _to_tensor(key: str, arr: np.ndarray) -> torch.Tensor:
    if key in _MASK_KEYS:
        return torch.as_tensor(arr.astype(np.bool_), dtype=torch.bool)
    if key in _INT_KEYS:
        return torch.as_tensor(arr.astype(np.int64), dtype=torch.long)
    return torch.as_tensor(arr.astype(np.float32), dtype=torch.float32)
