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

### Landed (iter-1, iter-2, iter-3)

- **Vectorization** (iter-1). `SubprocVecEnv` with 8 envs, one `RpcClient` per worker. Default in `train.py`.
- **Reward shaping** (iter-2). Floor-based terminal shaping `0.1 * (floor / 15) + 0.9 * is_win`, opt-in via `--reward-shape floor` (default on). Terminal-only reproduced by `--reward-shape none`.
- **Card-identity embedding** (iter-2, commit `3ed4d58`). `CARD_IDS` append-only tuple (67 engine ids + `UNKNOWN` sentinel) feeding a shared `nn.Embedding(D=32)`. `MaskableBacaPolicy` + `BacaFeaturesExtractor` apply masked single-head attention pooling over hand, shop, and reward card slots.
- **Attention-pool init + terminal-ReLU bugs** (iter-2). Both discovered and fixed during validation — see `doc/findings.md`.
- **Behavioral-cloning warm start** (iter-3). `src/baca/bc/` subpackage: Node-subprocess HeuristicBot bridge, NPZ-per-game dataset generator, streaming torch loader, masked cross-entropy trainer that reuses the frozen iter-2 policy, value-head pretrain, and `--bc-init` wiring in `train.py`. 50k HeuristicBot games at 29 games/sec, 3 BC epochs at batch 256 land 71.8% holdout imitation accuracy with 100% mask compliance.

### Outcome so far

Current best checkpoint is `checkpoints/v3-bc/bc_model.zip` — the BC-only snapshot, **1.2% winrate / 8.02 avg_floor** over 500 paired seeds. That is the first nonzero winrate for BACA (iter-2 plateaued at 0%) and a +1.01 avg_floor lift over iter-2's 7.01. HeuristicBot remains 3.56% / 10.23.

PPO fine-tune from the BC checkpoint (both BC-only and BC+value-pretrain variants, 200k timesteps, lr-scaled warmup + tight clip) **regressed** both metrics: BC→PPO landed 0.6% / 7.58, BC+value→PPO landed 0.0% / 7.33. Step 8.5's training traces show mean reward stuck in [0.04, 0.06] for both runs with no stable lift above BC's level. Details and failure-mode hypotheses in `doc/findings.md` "Iter-3 verdict" section. Iter-3 ships BC-only as the new best and deprecates the as-tuned PPO fine-tune until iter-4 revisits it.

### Remaining (iter-4+)

1. **Retune the PPO warm-start.** Iter-3 used `--total-timesteps 200000 --initial-lr-scale 0.33 --clip-range 0.1` and regressed BC-only. Cheapest next move is an A/B at shorter horizons (50k / 100k timesteps) with a tighter clip (0.05) and lower initial LR scale (0.1), to isolate whether it is length, step size, or architecture that erased the BC lift. Keep `checkpoints/v3-bc/bc_model.zip` frozen as the baseline.
2. **Larger BC dataset.** Iter-3 trained on 50k games → 3.1M samples → 71.8% holdout accuracy with the curve plateauing by epoch 3. 100k-200k games (~1-2h sustained collection at 29 g/s) may shift the ceiling, or may confirm it is policy-class-bound.
3. **LSTM or history-stacking policy.** `RecurrentPPO` from `sb3-contrib` addresses partial observability (hidden deck, enemy intent one turn ahead). Blocked on BC re-collection — history windows need to be captured during generation.
4. **Return-weighted BC loss.** Bias imitation toward wins by upsampling high-floor games during BC training. No new data; cheapest architecture-free lift.
5. **Larger network.** `features_dim=128`, `net_arch=[128, 128]`. Deferred — with BC anchoring the policy, this is less risky than it was in iter-2, but still likely dominated by (4) and (1).

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
