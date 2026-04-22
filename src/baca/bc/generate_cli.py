"""CLI wrapper for iter-3 BC dataset collection.

Drives ``collect_one_game`` in a sequential loop over a contiguous seed range.
Filenames use the decimal seed zero-padded to 8 digits (``seed_00000001.npz``).
Manifest entries accumulate incrementally so interrupted runs can be resumed
by re-launching with the same ``--out`` and ``--base-seed`` — the script skips
seeds that already appear in the manifest.

The output directory captures a ``CARD_IDS_snapshot.txt`` on first run so the
loader can reject datasets built against a drifted ``CARD_IDS`` tuple (R7).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from baca.bc.generator import (
    collect_one_game,
    sha256_of_file,
    write_game_npz,
)
from baca.bc.heuristic_bridge import HeuristicBridge
from baca.encoder import CARD_IDS
from baca.rpc_client import RpcClient

MANIFEST_NAME = "manifest.json"
SNAPSHOT_NAME = "CARD_IDS_snapshot.txt"
_PROGRESS_EVERY = 100


def _resolve_engine_dir() -> Path:
    override = os.environ.get("BACA_ENGINE_DIR")
    if override:
        return Path(override).resolve()
    return Path("../slay-the-ceper").resolve()


def _load_or_init_snapshot(out_dir: Path) -> None:
    snapshot_path = out_dir / SNAPSHOT_NAME
    current = json.dumps(list(CARD_IDS))
    if snapshot_path.is_file():
        recorded = snapshot_path.read_text(encoding="utf-8").strip()
        if recorded != current:
            raise ValueError(
                f"CARD_IDS drifted between collection runs; snapshot at {snapshot_path} does not "
                f"match current baca.encoder.CARD_IDS. Re-generate the dataset or pin the engine."
            )
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(current, encoding="utf-8")


def _load_manifest(out_dir: Path) -> list[dict[str, Any]]:
    manifest_path = out_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return []
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"manifest {manifest_path} is not a list")
    return raw


def _write_manifest(out_dir: Path, entries: list[dict[str, Any]]) -> None:
    manifest_path = out_dir / MANIFEST_NAME
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    tmp.replace(manifest_path)


def _seed_filename(seed: int) -> str:
    return f"seed_{seed:08d}.npz"


def run(
    out_dir: Path,
    games: int,
    base_seed: int,
    character_id: str,
    difficulty: str,
    engine_dir: Path | None = None,
) -> int:
    resolved_engine = engine_dir if engine_dir is not None else _resolve_engine_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    _load_or_init_snapshot(out_dir)

    manifest = _load_manifest(out_dir)
    already_done = {int(entry["seed"]) for entry in manifest}

    rpc = RpcClient(engine_dir=resolved_engine)
    bridge = HeuristicBridge(engine_dir=resolved_engine)
    collected = 0
    try:
        for offset in range(games):
            seed = base_seed + offset
            if seed in already_done:
                continue
            record = collect_one_game(
                rpc,
                bridge,
                seed,
                character_id=character_id,
                difficulty=difficulty,
            )
            filename = _seed_filename(seed)
            file_path = out_dir / filename
            write_game_npz(record, file_path)
            manifest.append(
                {
                    "seed": seed,
                    "filename": filename,
                    "n_steps": record["n_steps"],
                    "outcome": record["outcome"],
                    "floor_reached": record["floor_reached"],
                    "sha256": sha256_of_file(file_path),
                }
            )
            _write_manifest(out_dir, manifest)
            collected += 1
            if collected % _PROGRESS_EVERY == 0:
                print(  # noqa: T201 — CLI progress output
                    f"[baca.bc.generate_cli] {collected}/{games} games collected "
                    f"(last seed={seed}, outcome={record['outcome']}, "
                    f"floor={record['floor_reached']})",
                    flush=True,
                )
    finally:
        bridge.close()
        rpc.close()
    return collected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect BC dataset via HeuristicBot")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--games", type=int, required=True, help="Number of games to collect")
    parser.add_argument("--base-seed", type=int, default=1, help="First seed")
    parser.add_argument("--character-id", type=str, default="jedrek")
    parser.add_argument("--difficulty", type=str, default="normal")
    parser.add_argument(
        "--engine-dir", type=Path, default=None, help="Engine checkout (defaults to env override)"
    )
    args = parser.parse_args(argv)

    collected = run(
        out_dir=args.out,
        games=args.games,
        base_seed=args.base_seed,
        character_id=args.character_id,
        difficulty=args.difficulty,
        engine_dir=args.engine_dir,
    )
    print(  # noqa: T201 — CLI summary output
        f"[baca.bc.generate_cli] done — collected {collected} new game(s) into {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
