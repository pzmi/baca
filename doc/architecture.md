# BACA Architecture

```
┌──────────────────────────────────────┐           ┌───────────────────────────┐
│ BACA (Python)                        │   JSON    │ Usiec Cepra engine (Node) │
│                                      │   RPC     │                           │
│  MaskablePPO ── env.UsiecCepraEnv ───┼──────────►│  JsonRpcServer            │
│         ▲            │               │  stdio    │     ├── EngineController  │
│         │            ▼               │           │     │     (one per runId) │
│    encoder        rpc_client         │           │     └── RunRegistry       │
└──────────────────────────────────────┘           └───────────────────────────┘
```

## Roles

| Module            | Responsibility                                                                                                                                                          |
|:------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `baca.rpc_client` | Spawn `node scripts/rpc-server.js`, speak Content-Length framed JSON-RPC 2.0 on stdin/stdout, demux responses from server-pushed notifications.                         |
| `baca.encoder`    | Convert engine `Observation` JSON to a fixed-shape `gymnasium.spaces.Dict` of numpy arrays, including the legal-action mask.                                            |
| `baca.env`        | Gymnasium env wrapping one `runId`. `reset` disposes the old run and creates a fresh one. `step` looks up `legalActions[action_idx]` and forwards `engine.applyAction`. |
| `baca.train`      | MaskablePPO training loop with `ActionMasker` + `Monitor`.                                                                                                              |
| `baca.eval_cli`   | Load a checkpoint, play N episodes, report winrate + avg floor.                                                                                                         |

## Engine contract (summary)

Full spec: see [observation-action.md](./observation-action.md).

- **Determinism:** seeded PRNG (Mulberry32 in the engine); `(characterId, seed)` reproduces a run.
- **Action masking:** `observation.legalActions` is always populated until `done === true`. Illegal actions throw.
- **Phases:** `battle → map → reward | shop | campfire | event | maryna → battle → …`.
- **Termination:** `observation.done === true`; `outcome ∈ {player_win, enemy_win}` in the run summary.
- **Run lifecycle:** `engine.create` → `engine.startRun` → `engine.applyAction`× → `engine.dispose`.
- **Snapshots:** `engine.snapshot` + `engine.restore` exist; used later for AlphaZero-style tree search, not needed for v0 PPO.

## Process model

- **One Node subprocess per Python process.** A single `RpcClient` multiplexes up to 16 concurrent `runId`s (hard-coded `RUN_CAP`). For ≥16-env PPO, spawn multiple clients and round-robin run creation across them.
- **No batch RPC.** One request per message. Fine for PPO (sequential env steps per sub-env); a vectorised `VecEnv` would parallelize across workers, not within.
- **Notifications.** `engine.subscribe` is available for push events; v0 doesn't use it. Notifications land in `RpcClient.drain_notifications()` if ever subscribed.

## Why JSON-RPC over stdio and not HTTP

- Zero network setup; survives container boundaries.
- Framing is well-defined and already implemented server-side (LSP-style Content-Length).
- Subprocess lifecycle = env lifecycle; kill the Python process, `Popen` reaps the child.

## Repository layout

```
baca/
├── doc/                     # this documentation
├── src/baca/
│   ├── __init__.py
│   ├── rpc_client.py
│   ├── encoder.py
│   ├── env.py
│   ├── train.py
│   └── eval_cli.py
├── tests/
│   ├── unit/
│   │   ├── test_encoder.py
│   │   └── test_rpc_framing.py
│   └── integration/
│       └── test_smoke.py
├── pyproject.toml           # PDM + ruff + mypy + pytest config
└── README.md
```

Engine source of truth: <https://github.com/KlucasPL/slay-the-ceper>.
