# ML Approach

## Decision: Maskable PPO for v0

BACA starts as a **MaskablePPO** agent (Proximal Policy Optimization with
action masking, from `sb3-contrib`). This is not the fanciest option but it is
the right first step: mature library support, fast wall-clock training on a
sparse-reward, long-horizon problem, and trivial integration with the existing
`legalActions` from the engine.

## What was considered

| Approach                     | Fit    | Why / why not                                                                                                                                    |
|:-----------------------------|:-------|:-------------------------------------------------------------------------------------------------------------------------------------------------|
| **MaskablePPO (chosen)**     | High   | Policy-gradient; handles discrete masked actions natively; proven on card games (Big 2, Dominion); one library away from running.                |
| AlphaZero-style (MCTS + NN)  | High   | Engine already supports `snapshot`/`restore` and a `SearchBot` with MCTS; would give strong play. Deferred to phase 3 — needs a value net first. |
| DreamerV3 (model-based)      | Medium | Excellent on sparse-reward long-horizon tasks; complexity and compute cost outweigh benefits until PPO plateaus.                                 |
| DQN / Rainbow                | Medium | Off-policy, sample-efficient; less natural with variable action masks; weaker for high-variance trajectories common in roguelikes.               |
| NEAT / neuroevolution        | Low    | Simpler but empirically 10× slower wall-clock than PPO on comparable tasks; no GPU leverage due to arbitrary topologies.                         |
| Supervised on heuristic runs | Low    | Would cap BACA at heuristic strength (3.56% baseline). Useful only as a behavior-cloning warm start for PPO.                                     |
| LLM-as-agent                 | Low    | Interesting; orders of magnitude slower per step and token costs scale badly with thousands of games per training.                               |

## Why masked PPO specifically

1. **Native variable action set.** `legalActions` changes every step (hand size, phase). MaskablePPO's policy samples only over `env.action_masks() == True`, so no invalid-action penalty shaping is needed.
2. **On-policy, sample-abundant.** The engine runs at ~200 games/sec headless (HeuristicBot baseline). PPO's "needs lots of rollouts" weakness is cheap here.
3. **Mature tooling.** `stable-baselines3` + `sb3-contrib` give us logging, checkpointing, tensorboard, and evaluation callbacks for free.
4. **Clear next step.** Once a value net is trained well enough to beat HeuristicBot, it can be reused as the leaf evaluator in an AlphaZero-style MCTS variant.

## Reward design (v0)

Terminal-only:

```
reward = +1.0  if outcome == 'player_win'
reward =  0.0  otherwise
```

This is the cleanest signal but extremely sparse — the baseline winrate is
3.56%, so ~96% of episodes return zero. If PPO cannot learn anything in a few
hours of training, the next reward-shaping experiment is:

```
reward = 0.1 * (floor_reached / 15) + 0.9 * is_win
```

with the shaping term paid **only on termination**, not per-step. Per-step
rewards risk biasing the policy toward short-term plays that sacrifice
long-term deck-building.

### Iteration-2 update

Iteration-2 promotes the floor-based shaping to the default (`--reward-shape floor`),
still paid only on termination. Rationale: v0 training produced zero wins over 100k
steps; the value function had no signal to fit on ~96% zero-reward episodes, and the
research brief (`/.claude/research-brief-v2.md` §2 Tier-2(D)) flags sparse-reward
credit-assignment collapse as the root cause. Floor-progress shaping gives the critic
a graded target without introducing the per-step bias we were worried about in v0.
Truncations still pay zero reward — they are not terminal win/loss states.

If the shaped policy stalls at high survival / low wins (e.g. `avg_floor` improves
but winrate stays near zero), revert to `--reward-shape none` and revisit the
encoder or move to BC warm start.

## Hyperparameters (starting point)

```
learning_rate = 3e-4
n_steps       = 512       # per env per update
batch_size    = 64
gamma         = 0.99      # SB3 default
gae_lambda    = 0.95      # SB3 default
policy        = "MultiInputPolicy"   # for Dict observation space
```

Single env for v0; move to `SubprocVecEnv` once the smoke test is green and
convergence behavior is understood.

## Evaluation protocol

`baca-eval <checkpoint> --episodes 500` plays 500 seeded episodes against the
live engine and reports:

- `winrate`     — primary metric, compare to HeuristicBot's 3.56% (see `slay-the-ceper/baselines/main.metrics.json`).
- `avg_floor`   — depth reached; HeuristicBot averages ~10.2.
- `max_floor`   — best single run, sanity check for policy diversity.

A win-rate above HeuristicBot's is the bar to clear before claiming BACA beats
the baseline. Avg-floor improving without winrate improving usually means the
policy is surviving longer but failing to finish bosses.
