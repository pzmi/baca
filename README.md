# BACA

**B**ootstrapped **A**dvantage **C**ritic **A**gent — a reinforcement-learning agent for the
[Usiec Cepra](../slay-the-ceper) deckbuilder.

Named after the *baca*, the chief shepherd of a góralska watra. BACA learns to play Usiec Cepra
by self-play over the engine's JSON-RPC headless API.

## Stack

- Python 3.14 + PDM
- Gymnasium + Stable-Baselines3 + sb3-contrib (MaskablePPO)
- PyTorch

## Quick start

```bash
pdm install
pdm run test
pdm run train
```

`pdm run train` spawns `node scripts/rpc-server.js` from the sibling `slay-the-ceper` repo.
Set `BACA_ENGINE_DIR` to override the path.

## Documentation

See [`doc/`](./doc/) for architecture, observation/action spec, the chosen ML approach, and the
training roadmap.
