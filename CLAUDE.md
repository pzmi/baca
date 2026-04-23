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
- **Checkpoint stability.** `PHASES`, `CARD_TYPES`, `PLAYER_STATUSES`, and `CARD_IDS` in `encoder.py` are ordered tuples; any new entry must be **appended**, never inserted, or saved policies break. `CARD_IDS[0]` is the reserved `UNKNOWN` sentinel that absorbs engine cards not yet registered in the tuple.
- **Card-identity embedding** (iter-2, `src/baca/policy.py`). `BacaFeaturesExtractor` shares an `nn.Embedding(CARD_VOCAB_SIZE, card_embed_dim)` across hand / shop / reward blocks and attention-pools each block under its mask. Policy class is `MaskableBacaPolicy`.
- **BC data provenance** (iter-3, `src/baca/bc/`). `data/v3/CARD_IDS_snapshot.txt` is the trust boundary between collection and training; `BcDataset(root)` raises `ValueError` if it drifts from the current `encoder.CARD_IDS` tuple. Regenerate the dataset or pin the engine checkout — do not silently `touch` the snapshot.
- **PPO warm-start gotchas** (iter-3, `src/baca/train.py::_load_bc_model`). `MaskablePPO.load(bc_zip, env, custom_objects=...)` overrides only the three keys in `custom_objects` (`learning_rate`, `lr_schedule`, `clip_range`); every other scalar (`ent_coef`, `n_epochs`, `n_steps`, `batch_size`, `gamma`, `gae_lambda`) is restored from the BC zip's pickled state, where `save_bc_checkpoint` wrote the transient-training values (e.g. `n_steps=8, ent_coef=0.0`). `_load_bc_model` re-applies `cfg.*` after load and rebuilds `MaskableDictRolloutBuffer` because the original is materialized eagerly at load time.
- **Reward defaults to floor-shaped** (`--reward-shape floor`): `0.1*(floor/15) + 0.9*is_win` at termination, `0` on truncation. The pure-win signal is still available via `--reward-shape none`, but a 200k-step A/B on the current encoder (fixed init + no final ReLU) shows its value function collapses (`explained_variance` -12 to -34) under sparse terminal reward, whereas floor shaping keeps it healthy (+0.2 to +0.6) and lifts eval avg_floor from 2.01 (shape=none) to 7.01 (shape=floor). Don't add per-step shaping without updating `doc/ml-approach.md`.
- **Seeds.** `(characterId, seed)` reproduces a run. `env.reset(seed=...)` beats `base_seed`; `base_seed` is incremented per episode (`base_seed + episode_counter`) for variety.
- **RPC client is synchronous but threaded.** A reader thread demuxes responses by request `id`; notifications (from `engine.subscribe`) land in `drain_notifications()`. `dispose` failures are intentionally swallowed in `env.reset/close` — they are best-effort cleanup.

## Conventions specific to this repo

- **Test names:** `test_shouldResultWhenCondition` (camelCase after the `test_` prefix), with `# given` / `# when` / `# then` section comments. See `tests/integration/test_smoke.py`.
- **Python 3.14**, `ruff` line length 100, `mypy --strict`. Tests waive `S101`, `S105-7`, `S110`, `DTZ001`, `DTZ005` (see `pyproject.toml`).
- **Python 3.14 `except` syntax:** `except A, B:` (no parens) is valid and equivalent to `except (A, B):` — do NOT flag it as a Python 2 artifact. Confirmed via `ast.parse` and runtime in Python 3.14.3.
- **CLI `print` is allowed** via `# noqa: T201` on the specific line — do not disable `T20` globally.
- **Engine is the source of truth** for observation shape and action unions. If the engine's `Observation.js` or `ActionDispatcher.js` changes, update `doc/observation-action.md` and the encoder in the same commit.
