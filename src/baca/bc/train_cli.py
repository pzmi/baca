"""CLI entrypoint for BC training.

Runs ``train_bc`` over a ``BcDataset`` directory, saves the resulting policy
via :func:`save_bc_checkpoint`. Expected wall-clock for the iter-3 50k-game
dataset is 60-90 min on CPU (plan §3 Reconciliation 2, 5M-samples row).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from gymnasium import spaces

from baca.bc.dataset import BcDataset
from baca.bc.trainer import (
    EpochMetrics,
    build_bc_policy,
    freeze_value_head,
    save_bc_checkpoint,
    train_bc,
)
from baca.encoder import MAX_ACTIONS, observation_space


def _resolve_device(flag: str) -> torch.device:
    if flag == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(flag)


def run(
    data_dir: Path,
    out_path: Path,
    *,
    n_epochs: int,
    batch_size: int,
    lr: float,
    holdout_frac: float,
    seed: int,
    device: str,
    card_embed_dim: int = 32,
    features_dim: int = 64,
) -> list[EpochMetrics]:
    dataset = BcDataset(data_dir)
    obs_space = observation_space()
    act_space: spaces.Space[Any] = spaces.Discrete(MAX_ACTIONS)
    policy_kwargs: dict[str, Any] = {
        "card_embed_dim": card_embed_dim,
        "features_dim": features_dim,
    }
    policy = build_bc_policy(obs_space, act_space, policy_kwargs=policy_kwargs)
    frozen = freeze_value_head(policy)
    print(  # noqa: T201 — CLI summary output
        f"[bc.train_cli] dataset={data_dir} games={dataset.game_count} "
        f"samples={len(dataset)} frozen_value_params={frozen}",
        flush=True,
    )

    torch_device = _resolve_device(device)
    metrics = train_bc(
        policy,
        dataset,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        holdout_frac=holdout_frac,
        seed=seed,
        device=torch_device,
    )
    policy.to(torch.device("cpu"))
    save_bc_checkpoint(policy, obs_space, act_space, out_path, policy_kwargs=policy_kwargs)
    print(  # noqa: T201 — CLI summary output
        f"[bc.train_cli] saved checkpoint to {out_path}", flush=True
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Behavioral-cloning trainer for BACA")
    parser.add_argument("--data", type=Path, required=True, help="BC dataset directory")
    parser.add_argument("--out", type=Path, required=True, help="Output checkpoint zip path")
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--holdout-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Torch device (auto picks cuda if available).",
    )
    parser.add_argument("--card-embed-dim", type=int, default=32)
    parser.add_argument("--features-dim", type=int, default=64)
    args = parser.parse_args(argv)

    run(
        data_dir=args.data,
        out_path=args.out,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        holdout_frac=args.holdout_frac,
        seed=args.seed,
        device=args.device,
        card_embed_dim=args.card_embed_dim,
        features_dim=args.features_dim,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
