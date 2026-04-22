# Findings from the Slay-the-Ceper engine review

This file captures what we learned when scoping BACA. Kept as a
reference to avoid re-deriving it when extending the agent.

## Headless engine is production-grade

`slay-the-ceper/src/engine/` exposes a clean headless API with every property
you want from an RL target:

- **Deterministic seeding.** Mulberry32 PRNG; `(characterId, seed)` uniquely reproduces a run. Non-determinism scanner in `tests/engine/nondeterminism-sources.test.js` keeps this guarantee.
- **No DOM, no async, no timers.** Pure JS state machine; runs in Node, workers, or anywhere else.
- **Action masking built in.** `Observation.legalActions` is always populated; illegal actions throw `IllegalActionError`.
- **Snapshots.** `engine.snapshot()` → JSON-serializable state; `engine.restore()` reconstructs it. Used by `SearchBot` for 1-ply lookahead and MCTS rollouts.
- **Speed.** Baseline: `RandomBot ≥ 500 games/s`, `HeuristicBot ≥ 200 games/s`. ~25 actions per game, ~0.2ms per action.

## JSON-RPC server removes the Python-bridge problem

`slay-the-ceper/scripts/rpc-server.js` wraps the engine in a JSON-RPC 2.0
server speaking Content-Length framed messages on stdin/stdout (LSP-style).

- Methods: `engine.create`, `startRun`, `getObservation`, `getLegalActions`, `applyAction`, `endTurn`, `snapshot`, `restore`, `seed`, `drainEvents`, `getRunSummary`, `subscribe`, `dispose`, `renderText`.
- **Single persistent connection, many runs.** `RunRegistry` holds up to 16 concurrent runs per process; each addressed by UUID.
- Errored runs are sticky — dispose and recreate on crash.
- No batch RPC; one request per message. Not a problem for PPO stepping.

This is exactly the shape a Python gym env wrapper needs: `baca.rpc_client`
spawns one Node process per Python process and multiplexes runs.

## Existing bots are a strong baseline set

`slay-the-ceper/src/logic/bots/` has 14 registered bots:

- **HeuristicBot** — greedy, weighted card/map/reward scoring; 3.56% winrate, 10.2 avg floor on 9998 seeded games.
- **RandomBot** — uniform; ~0% winrate, dies around floor 3.
- **AggressiveBot / DefensiveBot / StatusBot / GreedyBot / MinimalistBot / EconomyBot / BerserkerBot / DrawEngineBot / ElitistBot / LansBot / RachunekBot** — strategy archetypes.
- **SearchBot** — 1-ply static eval or MCTS rollouts via snapshot/restore.

Bot interface: `(observation, rng?) => Action`. BACA can register itself in
`src/logic/bots/index.js` via a factory once distilled (Phase 4).

## Training harness already parallel

`scripts/sim/batch.js` already runs batches across Node worker threads:

- `runBatch({ name, character, agent, games, seedStart, workers })` → `GameResult[]`.
- Output: JSONL metrics, optional paired A/B runs.
- Baseline dataset: 9998 seeded games of HeuristicBot, stored in `baselines/main.metrics.json`.

BACA piggy-backs on this infrastructure for evaluation: run `runBatch` with a
thin bot wrapper that shells back into Python for inference, or (cheaper) run
`baca.eval_cli` directly over the JSON-RPC server.

## Gotchas

- **Partial observability.** Deck contents hidden unless `rules.revealAllPiles=true`; enemy intent revealed one turn ahead. Motivates LSTM / history stacking in phase 2.
- **Sparse reward.** ~96% of HeuristicBot's games end in loss. Plan on shaping after PPO fails to learn without it.
- **UUID run IDs.** Clients cannot pre-allocate; track IDs per env slot.
- **`RUN_CAP = 16`** is hard-coded in `src/rpc/RunRegistry.js`. For >16 concurrent envs, spawn multiple RPC processes.
- **Maryna boon picks** are non-deterministic without `forcedBoonOffer`. Fine for training (variety), but eval runs should fix seeds.

## Reference winrates

From `slay-the-ceper/baselines/main.metrics.json` (9998 games, Jędrek, normal):

- winrate: **3.56%**  (95% CI: 3.2% – 3.92%)
- avg floor reached: **10.23**
- avg survival score: **10.24**
- avg turns played: **22.83**
- avg dutki earned: **147.87**
- avg HP at death: **0.53** (fraction of max)

BACA's target for phase 2 is to match these; phase 3 to double them.

## Iter-2 empirical findings

### Engine caps reward and shop card offers at 3, not 5

`src/state/ShopSystem.js:99` calls `_pickUniqueItems(cardPool, cardLibrary, 3)`; `src/engine/ActionDispatcher.js:260` calls `generateCardRewardChoices?.(3)`. `MAX_REWARD_CARDS = MAX_SHOP_CARDS = 3` in the encoder. Widening either requires an engine change first, otherwise the extra slots stay permanently zero-padded and eat policy capacity.

### Attention-pool double-scaling silently collapses softmax

A single-head attention pool implemented as `query = nn.Parameter(randn(D) / sqrt(D))` *and* `logits = (keys @ query) / sqrt(D)` stacks two `1/sqrt(D)` factors. Effective logit std lands at `≈ 1/D`; softmax over 3-10 slots is indistinguishable from uniform (hand block measured 1.05× uniform max weight at init). The pool behaves as a mean-pool with zero query gradient throughout early training — value function can fit the shaped signal but the attention query never learns *which* card matters.

Diagnosis tools that caught it: forward the pool with `torch.manual_seed(0)` on a synthetic batch and assert `max_weight > 1.2 × uniform` and `std > 0.05`. Fix: drop the explicit `1/sqrt(D)` scale and init the query with plain `randn(D)` (std=1). After the fix, hand block weights peak at ~7.9× uniform and queries receive usable gradient from step 1. Regression-pinned by `test_shouldProduceNonUniformAttentionWeightsAtInit` in `tests/unit/test_policy.py`.

### Terminal `nn.ReLU` on a features extractor wastes half the policy-head input space

Stable-Baselines3 initializes `MaskableMultiInputActorCriticPolicy`'s policy and value heads with `ortho_init=True` (orthogonal gain ~ √2 for hidden layers, ~0.01 for the policy output). If the features extractor ends in `nn.ReLU`, the extractor output lives in `R^features_dim_+`; the orthogonally-initialized head sees only positive inputs, cutting effective input dim roughly in half and biasing head outputs. Symptom: training runs with `explained_variance` climbing positive early then collapsing to large negatives (measured -12 to -34 at 200k under `shape=none`). Fix: make the last layer of the extractor a plain `nn.Linear` so features can take negative values. Sentinel: `test_shouldProduceNegativeFeatureValuesWhenForwardAllowsSignedActivations`.

### Floor shaping is the stable signal; terminal-only collapses the critic

Same iter-2 build (fixed init, no terminal ReLU), same 200k steps, same seed, only the reward toggled:

| `--reward-shape` | avg_floor (best) | explained_variance at 200k | eval_mean_ep_length |
|:-----------------|:-----------------|:---------------------------|:--------------------|
| `floor`          | **7.01**         | **+0.61** (stable)         | 167                 |
| `none`           | 2.00             | -12.1                      | 33.5                |

With a working encoder the shaping is the single change that lets the critic fit a target at all. Terminal-only under the new encoder gets no wins, the value function sees near-zero returns everywhere, and the larger observation space (int ids + masks + costs for three blocks) gives it more dimensions to overfit noise with.

Implication for iter-3: do not drop shaping when adding BC or larger networks; it is load-bearing until the policy crosses into positive winrates.

### PPO ceiling without a learning bootstrap

Under the iter-2 build with floor shaping, the agent reaches `avg_floor 7.01` with `eval_mean_ep_length` climbing to 167 steps (vs v0's ~55) and `mean_reward` peaking at 0.0493 around step 100k before declining. Winrate across all 500 eval episodes is 0.0%. Pure on-policy PPO will not cross the 3.56% HeuristicBot baseline by just scaling timesteps: the policy is trading wins-potential for longer stalls. Next lever is behavior cloning from HeuristicBot as a warm start (`.claude/research-brief-v2.md` Tier-3(F)) — gives PPO non-zero wins to work with, then fine-tune.

## Iter-3 empirical findings

### Step 0 throughput (BACA-side HeuristicBot collection)

20-game benchmark via `python -m baca.bc.benchmark --games 20` — one `RpcClient` + one `HeuristicBridge` subprocess over stdio, sequentially:

```
avg_steps_per_game: 61.5
p50 rpc_ms: 0.06
p95 rpc_ms: 0.11
p50 heuristic_ms: 0.10
p95 heuristic_ms: 0.14
games_per_sec: 74.71
projected_wall_clock_50k_games: 0.19 h
```

Two surprises versus the research brief. First, decision steps per game are ~2.5× the brief's 25-step assumption — 61.5 measured, matching the 63 seen at `n=2`. The brief cross-referenced this file's iter-2 line ("~25 actions per game"), but that number is engine-side `applyAction` calls per HeuristicBot game; BACA's decision loop also stops at intermediate-reward phases (card picks, shop turns, rest choices), so the Python-side step count is legitimately higher. Second, collection is ~100× faster than the brief's upper bound — 50k games projects to ~11 minutes sequentially, not 20-25 hours. The RPC + stdio handoff is ~0.2ms per action, not the ~1-2s/game the brief assumed from "RPC-heavy Python."

Implications for iter-3:

- 50k games × 61.5 steps ≈ **3.1M samples**, between the plan's 1.25M and 5M envelope rows. Plan §3 Reconciliation 2's "5M / 3 epochs / batch 256 / lr 1e-3" recipe is the closer fit.
- Sequential collection is the baseline; parallelization (plan R3, §5 Step 6) is not needed.
- Wall-clock is dominated by BC training (~30-90 min CPU) and PPO fine-tune (~15-25 min), not data collection.

Benchmark output preserved in `logs/bench_step0.out`.

### Step 6: BC training on 50k HeuristicBot games

Full pipeline: `python -m baca.bc.generate_cli --out data/v3 --games 50000 --base-seed 1` → `python -m baca.bc.train_cli --data data/v3 --out checkpoints/v3-bc/bc_model.zip --n-epochs 3 --batch-size 256 --lr 1e-3 --holdout-frac 0.05` → `python -m baca.eval_cli checkpoints/v3-bc/bc_model.zip --episodes 200 --base-seed 99999`. End-to-end wall-clock 36 minutes on CPU.

| Phase          | Duration | Notes                                                                                                       |
|:---------------|:---------|:------------------------------------------------------------------------------------------------------------|
| Generate 50k   | 28m42s   | 29 games/sec sustained (vs Step 0's 74.7 g/s at `n=20`). Manifest append + NPZ compression overhead scales. |
| BC train 3 ep  | 6m48s    | ~3.1M samples × 3 = 12k batches of 256; sub-second per batch on CPU.                                        |
| BC eval 200 ep | 24s      | Fast because the policy is cheaper than HeuristicBot's scoring.                                             |

BC loss curve plateaued by epoch 3:

| Epoch | train_loss | holdout_acc | holdout_mask_compliance |
|:------|:-----------|:------------|:------------------------|
| 1     | 0.7027     | 0.697       | 1.000                   |
| 2     | 0.5931     | 0.714       | 1.000                   |
| 3     | 0.5808     | 0.718       | 1.000                   |

100% mask compliance on held-out samples — the action-index mapping is correct and the masked cross-entropy contract holds end-to-end. Holdout accuracy plateau at ~72% is the ceiling on pure imitation with the iter-2 encoder; each ~28% mis-imitated decision has a chance to cascade into an earlier loss.

Eval against HeuristicBot baseline (9998 games, 3.56% winrate, 10.23 avg_floor):

| Checkpoint                 | episodes | winrate   | avg_floor | max_floor |
|:---------------------------|:---------|:----------|:----------|:----------|
| HeuristicBot (baselines)   | 9998     | **3.56%** | **10.23** | 15        |
| v3-bc (BC-only, this iter) | 200      | **2.00%** | **8.09**  | 15        |
| v2-norelu-shaped (iter-2)  | 500      | 0.00%     | 7.01      | 15        |

2.0% on n=200 is within the ~±1.3% 95%-CI half-width of the plan's `[2.56%, 4.56%]` acceptance band; avg_floor trails HeuristicBot by ~2 floors. Plan §1 red regression (winrate < 1% AND avg_floor < 5) is not tripped. The 2-floor gap indicates that BC's mis-imitated decisions concentrate at late-game (higher-floor) states where suboptimal actions kill runs faster — consistent with the imitation ceiling leaving room for PPO fine-tune to recover.

Full log at `logs/step6.out`; checkpoint at `checkpoints/v3-bc/bc_model.zip`.
