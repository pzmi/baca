# Observation & Action Specification

Source of truth: `slay-the-ceper/src/engine/Observation.js` and
`slay-the-ceper/src/engine/ActionDispatcher.js`.

## Engine observation (JSON)

```ts
Observation = {
  phase: 'battle' | 'map' | 'reward' | 'shop' | 'campfire' | 'event' | 'maryna',
  turn: number,            // total turns played across the run
  battleTurn: number,      // turns elapsed in the current combat
  floor: number,           // 1-indexed (map level + 1)
  act: number,             // 1..3
  done: boolean,
  outcome?: 'player_win' | 'enemy_win',

  weather: { id: string, name: string, description: string },

  player: {
    hp: number, maxHp: number, block: number,
    energy: number, maxEnergy: number,
    status: StatusDef,     // weak, vulnerable, fragile, strength, lans, duma_podhala, furia_turysty, next_double, ...
    stunned: boolean,
    cardsPlayedThisTurn: number,
  },

  enemy: {
    id, name, hp, maxHp, block, status, passive,
    isElite, isBoss, rachunek, ped,
    intent: { type, name, hits, expectedDamageToPlayer, text },
    phaseTwoTriggered, stunnedTurns, evasionCharges,
    bossArtifact?,
  } | null,

  hand: CardView[],        // { id, name, type, cost, effectiveCost, desc, emoji, unplayable, exhaust, tags }
  deckCount, discardCount, exhaustCount,

  combat: { firstAttackUsed, activeSide, attackCardsPlayedThisBattle },

  run: {
    character, difficulty, dutki, relics,
    marynaBoon, cardDamageBonus,
    acquired: { cards, relics, boons },
  },

  legalActions: Action[],   // always non-empty unless done

  // Phase-conditional fields
  map?: { currentLevel, currentNodeIndex, totalLevels, currentNode, reachableNodes },
  activeEvent?: { id, name, description, choices: { index, text }[] } | null,
  shopStock?: { cards, relic },
  campfire?: { upgradeable: string[] },
  marynaOffer?: string[] | null,
  rewardOffer?: object | null,

  fullDeck?, discardContents?, exhaustContents?,  // only if rules.revealAllPiles
}
```

Observations are **deep-frozen**; safe to keep as trace data without copying.

## Engine actions (union)

```ts
// Combat
{ type: 'play_card',   handIndex: number }
{ type: 'end_turn' }
{ type: 'smycz_toggle', handIndex: number | null }

// Map
{ type: 'travel', level: number, nodeIndex: number }

// Reward
{ type: 'reward_pick_card', cardId: string | null }  // null = skip
{ type: 'reward_pick_relic', relicId: string }

// Shop
{ type: 'shop_buy_card',    cardId: string }
{ type: 'shop_buy_relic',   relicId: string }
{ type: 'shop_remove_card', cardId: string }
{ type: 'shop_leave' }

// Campfire
{ type: 'campfire', option: 'rest' | 'leave' }
{ type: 'campfire', option: 'upgrade', cardId: string }

// Event
{ type: 'event_choice', choiceIndex: number }

// Maryna
{ type: 'maryna_pick', boonId: string }
```

Each `applyAction` is **atomic**: one card play, one travel, one shop purchase. No multi-action sequences.

## BACA encoded observation (v0)

`baca.encoder.encode(obs)` returns a `gymnasium.spaces.Dict`:

| Key             | Space                                     | Contents                                                                                                                                              |
|:----------------|:------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------|
| `player`        | `Box(4 + len(PLAYER_STATUSES),)`          | `[hp/maxHp, block/50, energy/maxEnergy, cardsPlayedThisTurn/10, weak, vulnerable, fragile, strength, lans, duma_podhala, furia_turysty, next_double]` |
| `enemy`         | `Box(5,)`                                 | `[hp/maxHp, block/50, intentDmg/playerMaxHp, isElite, isBoss]` — zeros if out of battle                                                               |
| `enemy_present` | `Box(1,)`                                 | `1.0` if `obs.enemy != null` else `0.0`                                                                                                               |
| `phase`         | `Box(len(PHASES),)`                       | one-hot over `{battle, map, reward, shop, campfire, event, maryna}`                                                                                   |
| `run`           | `Box(4,)`                                 | `[floor/15, act/3, dutki/200, deckCount/30]`                                                                                                          |
| `hand`          | `Box(MAX_HAND_SIZE, 2 + len(CARD_TYPES))` | per card: `[cost/3, isPlayable, one-hot(type)]`; padded with zeros                                                                                    |
| `hand_mask`     | `MultiBinary(MAX_HAND_SIZE)`              | `1` for real cards, `0` for padding                                                                                                                   |
| `action_mask`   | `MultiBinary(MAX_ACTIONS)`                | `1` for indices `< len(legalActions)`                                                                                                                 |

Constants (`baca.encoder`):

```
MAX_HAND_SIZE = 10
MAX_ACTIONS   = 32
PHASES         = (battle, map, reward, shop, campfire, event, maryna)
CARD_TYPES     = (attack, skill, power, status, curse)
PLAYER_STATUSES = (weak, vulnerable, fragile, strength, lans, duma_podhala, furia_turysty, next_double)
```

## BACA action space (v0)

`gymnasium.spaces.Discrete(MAX_ACTIONS)`. Index `i` maps to
`observation.legalActions[i]`. `MaskablePPO` reads `env.action_masks()` to
sample only from valid indices.

Out-of-mask indices are clamped to `0` as a defensive fallback — `MaskablePPO`
should never emit them when the mask is correctly wired.

## Known limitations (v0)

- **No card-identity embedding.** Cards are encoded by type + cost only; two cards of the same type look identical to the policy.
- **Partial observability not modelled.** No LSTM / history stacking; policy sees a single frame. Fine for combat-tactical play, weaker for multi-turn planning.
- **No event / shop content encoding.** The policy can pick an event choice or shop slot by index, but doesn't see the contents directly (only phase one-hot).
- **Max action count capped at 32.** Enough for all current phases (hand of 10 + end_turn + smycz is 12; events typically ≤ 4 choices). Raise if new content exceeds.
