# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

BACA is a Python RL agent (MaskablePPO) that plays the **Usiec Cepra** deckbuilder by driving its
headless Node engine over JSON-RPC. The engine lives in a sibling repo `slay-the-ceper` (same parent
directory by default; override with `BACA_ENGINE_DIR`). BACA itself contains no game logic — it is a
gymnasium wrapper + training/eval loop.

The RL approach and long-term plan are already documented under `doc/` (`architecture.md`,
`observation-action.md`, `ml-approach.md`, `roadmap.md`, `findings.md`). Read those before changing
encoder shape, reward, or algorithm choice — decisions and tradeoffs are captured there.

## Common commands (PDM)

- `pdm install` — install deps into `.venv`.
- `pdm run test` — pytest over `tests/unit` and `tests/integration`.
- `pdm run test-cov` — coverage report (`term-missing` + HTML).
- `pdm run lint` — `ruff check` + `mypy src/ tests/` (mypy is `strict`).
- `pdm run fmt` — `ruff format`.
- `pdm run check` — fmt + lint + tests, the pre-commit gate.
- `pdm run train` — runs `python -m baca.train` with `PYTHONPATH=src`; spawns `node scripts/rpc-server.js` in `$BACA_ENGINE_DIR` (default `../slay-the-ceper`).
- Console scripts installed by the project: `baca-train`, `baca-eval <checkpoint>`.
- Run one test: `pdm run pytest tests/unit/test_encoder.py::test_name -vv`.

The integration smoke test (`tests/integration/test_smoke.py`) **auto-skips** when the engine
checkout is missing, so a green local run does not prove the RPC path works — point
`BACA_ENGINE_DIR` at the engine to actually exercise it.

## Architecture in one picture

```
MaskablePPO ── UsiecCepraEnv ── RpcClient ──stdio JSON-RPC──►  node scripts/rpc-server.js
                    │                                                  (engine + RunRegistry)
                 encoder
```

Module responsibilities (`src/baca/`):

| Module       | Role                                                                         |
|:-------------|:-----------------------------------------------------------------------------|
| `rpc_client` | One Node subprocess; Content-Length framed JSON-RPC 2.0; thread-safe `call`. |
| `encoder`    | Engine `Observation` JSON → fixed-shape `spaces.Dict` + legal-action mask.   |
| `env`        | `gymnasium.Env` owning one `runId`; `reset` disposes + recreates the run.    |
| `train`      | `MaskablePPO` + `ActionMasker` + `Monitor`; single env for v0.               |
| `eval_cli`   | Load checkpoint, play N games, print winrate/avg_floor/max_floor.            |

## Non-obvious constraints

- **Engine `RUN_CAP = 16`** per Node process. One `RpcClient` owns one subprocess; for >16 concurrent envs (e.g. `SubprocVecEnv`), spawn multiple clients and round-robin run creation.
- **Action space is fixed** `Discrete(MAX_ACTIONS=32)`; index `i` maps to `observation.legalActions[i]`. Out-of-mask indices are clamped to `0` in `env.step` as a defensive fallback — if you change the mask wiring, revisit that clamp.
- **Checkpoint stability.** `PHASES`, `CARD_TYPES`, `PLAYER_STATUSES` in `encoder.py` are ordered tuples; any new entry must be **appended**, never inserted, or saved policies break.
- **No card-identity embedding** in v0 (type + cost only). Same-type/cost cards are indistinguishable to the policy — expected, deferred to phase 2 per `doc/roadmap.md`.
- **Reward is terminal-only** (`+1` on `player_win`, else `0`). Baseline HeuristicBot winrate is 3.56%, so ~96% of episodes carry zero signal. Don't add per-step shaping without updating `doc/ml-approach.md`.
- **Seeds.** `(characterId, seed)` reproduces a run. `env.reset(seed=...)` beats `base_seed`; `base_seed` is incremented per episode (`base_seed + episode_counter`) for variety.
- **RPC client is synchronous but threaded.** A reader thread demuxes responses by request `id`; notifications (from `engine.subscribe`) land in `drain_notifications()`. `dispose` failures are intentionally swallowed in `env.reset/close` — they are best-effort cleanup.

## Conventions specific to this repo

- **Test names:** `test_shouldResultWhenCondition` (camelCase after the `test_` prefix), with `# given` / `# when` / `# then` section comments. See `tests/integration/test_smoke.py`.
- **Python 3.14**, `ruff` line length 100, `mypy --strict`. Tests waive `S101`, `S105-7`, `S110`, `DTZ001`, `DTZ005` (see `pyproject.toml`).
- **CLI `print` is allowed** via `# noqa: T201` on the specific line — do not disable `T20` globally.
- **Engine is the source of truth** for observation shape and action unions. If the engine's `Observation.js` or `ActionDispatcher.js` changes, update `doc/observation-action.md` and the encoder in the same commit.
