"""CLI wrapper for iter-3 Step 7 value-head pretraining.

Loads a BC checkpoint, freezes the policy side, fits the value head on MC
returns computed from the collected HeuristicBot dataset, and saves a new
SB3-compatible checkpoint for Step 8's PPO warm-start.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gymnasium import spaces
from sb3_contrib import MaskablePPO

from baca.bc.dataset import BcDataset
from baca.bc.trainer import _make_dummy_vec_env
from baca.bc.value_pretrain import (
    EpochMetrics,
    freeze_policy_side,
    save_value_checkpoint,
    train_value_head,
)
from baca.encoder import MAX_ACTIONS, observation_space


def _summarize_epoch(m: EpochMetrics) -> str:
    return (
        f"epoch={m.epoch} train_loss={m.train_loss:.5f} "
        f"pred_mean={m.val_pred_mean:.4f} pred_std={m.val_pred_std:.4f} "
        f"returns_mean={m.returns_mean:.4f} returns_std={m.returns_std:.4f}"
    )


def run(
    bc_checkpoint: Path,
    data_root: Path,
    out_path: Path,
    *,
    gamma: float,
    n_epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
) -> int:
    obs_space = observation_space()
    action_space: spaces.Space[int] = spaces.Discrete(MAX_ACTIONS)
    vec_env = _make_dummy_vec_env(obs_space, action_space)
    try:
        model = MaskablePPO.load(str(bc_checkpoint), env=vec_env)
        policy = model.policy
        frozen = freeze_policy_side(policy)  # type: ignore[arg-type]
        trainable = sum(1 for p in policy.parameters() if p.requires_grad)
        print(  # noqa: T201 — CLI summary output
            f"[value-pretrain] loaded {bc_checkpoint} — frozen={frozen} trainable={trainable}",
            flush=True,
        )

        dataset = BcDataset(data_root)
        print(  # noqa: T201 — CLI summary output
            f"[value-pretrain] dataset root={dataset.root} "
            f"games={dataset.game_count} samples={len(dataset)}",
            flush=True,
        )

        metrics = train_value_head(
            policy,  # type: ignore[arg-type]
            dataset,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            gamma=gamma,
            seed=seed,
            device=device,
        )

        for m in metrics:
            print(f"[value-pretrain] {_summarize_epoch(m)}")  # noqa: T201

        policy_kwargs = dict(model.policy_kwargs) if model.policy_kwargs else None
        save_value_checkpoint(
            policy,  # type: ignore[arg-type]
            obs_space,
            action_space,
            out_path,
            policy_kwargs=policy_kwargs,
        )
        print(f"[value-pretrain] saved to {out_path}")  # noqa: T201
    finally:
        vec_env.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pretrain value head on MC returns from a BC dataset"
    )
    parser.add_argument("--bc-checkpoint", type=Path, required=True, help="BC checkpoint zip")
    parser.add_argument("--data", type=Path, required=True, help="BC dataset root")
    parser.add_argument("--out", type=Path, required=True, help="Output checkpoint zip")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)

    return run(
        bc_checkpoint=args.bc_checkpoint,
        data_root=args.data,
        out_path=args.out,
        gamma=args.gamma,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    sys.exit(main())
