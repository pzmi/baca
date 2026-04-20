# Roadmap

Four phases, each gated by a concrete success metric against the HeuristicBot
baseline (3.56% winrate, 10.2 avg floor on `jedrek` / normal).

## Phase 0 — Bootstrap (this commit)

**Goal:** Pipeline runs end-to-end. Random policy plays a full game through
the Python wrapper without crashing.

Artifacts:

- `baca.rpc_client` connects to `node scripts/rpc-server.js`.
- `baca.env` exposes a gymnasium env with action masking.
- `baca.encoder` produces fixed-shape observations.
- `baca.train` trains MaskablePPO for a configurable number of timesteps.
- `baca.eval_cli` measures winrate and avg floor.
- Smoke test plays one random episode end-to-end.

**Done when:** `pdm run pytest tests/` passes (unit + integration once the
engine checkout is present next door).

## Phase 1 — Beat RandomBot

**Goal:** PPO checkpoint with > 1% winrate on 500 seeded games. Low bar, but
proves the encoder and reward are pointing the policy in the right direction.

Focus:

- 100k–500k training timesteps.
- Single env, `MultiInputPolicy` with default MLP extractor.
- Terminal reward only; no shaping.

**Done when:** `baca-eval --episodes 500` reports winrate above RandomBot's
~0% and avg floor above ~5 (RandomBot dies around floor 3).

## Phase 2 — Match HeuristicBot

**Goal:** Match or exceed 3.56% winrate and 10.2 avg floor.

Experiments (in priority order):

1. **Reward shaping** — add `0.1 * (floor / 15)` at termination.
2. **Vectorization** — `SubprocVecEnv` with 8–16 envs (4 RPC processes × 4 runs each).
3. **Larger network** — widen MLP to 256/256 or add a small transformer over the hand encoding.
4. **LSTM policy** — `RecurrentPPO` from `sb3-contrib` to model partial observability (hidden deck order, future draws).
5. **Card-identity embedding** — replace the type-only encoding with a learned embedding over card IDs.

**Done when:** Paired eval over 1000 seeds shows BACA winrate confidence
interval overlapping or exceeding HeuristicBot's.

## Phase 3 — Beat HeuristicBot decisively

**Goal:** 2×+ HeuristicBot winrate on normal, and a non-trivial winrate on
hard difficulty.

Options (likely combined):

- **Self-play opponents.** Freeze an older BACA as the "enemy" curriculum stage.
- **AlphaZero-style MCTS.** Swap the HeuristicBot leaf evaluator in the existing `SearchBot` for a BACA value head. Use `engine.snapshot` / `engine.restore` for tree expansion over RPC.
- **Population-based training.** Keep a small population of checkpoints with different hyperparams; periodically copy the best weights over the worst.

**Done when:** >= 10% winrate on normal and >= 3% on hard, stable across seeds.

## Phase 4 — Ship distilled in-game bot

**Goal:** Distill BACA's policy into a lightweight in-game opponent (JUHAS).

Tasks:

- Export BACA policy to ONNX.
- Port the runtime to JS (ONNX Runtime Web or a hand-rolled MLP executor) so the game can ship it without Python.
- Integrate with `src/logic/bots/` in the engine repo alongside HeuristicBot and SearchBot.

**Done when:** The distilled bot registers in `src/logic/bots/index.js` and
runs in the game's existing bot benchmarks without Python in the loop.

## Non-goals (explicit)

- **Hard difficulty optimization** before matching HeuristicBot on normal.
- **Multi-character training.** Start with `jedrek` only; expand once the pipeline works.
- **Imitation learning from human runs.** No human trace dataset exists; not worth building one yet.
- **Hyperparameter sweeps.** Avoid until a single configuration shows signs of learning.
