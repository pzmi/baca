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
