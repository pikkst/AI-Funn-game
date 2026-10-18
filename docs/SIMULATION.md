# Survival-v2 world and environment specification

**Status:** design contract for M1, subject to explicit versioning after implementation.

## World

- Discrete 32×32 tile grid with terrain: passable ground, impassable water, traversable forest and food-bearing tiles.
- A seeded, configurable procedural generator creates valid reachable spawn positions and sufficient navigable terrain; reject/regenerate impossible layouts with bounded retries.
- One organism per environment; orientation-free four-directional movement; no other agents in M1.
- Resource placement, regrowth and consumption are governed by deterministic, tested rules.
- Use a finite observation window: 7×7 centered on agent; cells outside map are encoded as impassable.
- Generating a map must not guarantee a successful episode or reveal hidden resources to the agent.

## State

`WorldState`: terrain, food resources, organism coordinate, energy, health, hunger, current step, step-local event list and environment-owned RNG state.

Initial nominal values and dynamics must be centralized in a versioned `SurvivalConfig` (do not spread constants through modules). Proposed defaults, to calibrate by baseline tests:

| Parameter | Initial proposal |
|---|---|
| World dimensions | 32 × 32 |
| Visible area | 7 × 7 |
| Max episode steps | 512 |
| Energy | 100 normalized to [0,1] |
| Health | 100 normalized to [0,1] |
| Hunger | 0 = satiated; increases over time |
| Food units per resource tile | configurable |
| Step metabolism | consumes energy every step |
| Rest | recovers energy subject to hunger/food constraints |
| Eat | consumes resource only on current tile |

Energy restoration must never permit indefinite survival without consuming food: persistent metabolic/hunger costs eventually cause death in resource-starved worlds. Explicitly test an always-rest policy.

## Action mapping (frozen for survival-v2)

| ID | Action | Semantics |
|---|---|---|
| 0 | MOVE_NORTH | attempt y - 1 |
| 1 | MOVE_SOUTH | attempt y + 1 |
| 2 | MOVE_WEST | attempt x - 1 |
| 3 | MOVE_EAST | attempt x + 1 |
| 4 | EAT | consume food at current position if present |
| 5 | REST | remain in place; bounded recovery |

Invalid movement leaves coordinate unchanged but still incurs step cost; failed EAT also incurs cost. Never turn invalid actions into free exploration or free energy.

## Observation specification

A fixed `float32[199]`: 7×7×4 spatial channels + 3 internal values.

Four spatial channels, row-major relative to agent:
1. passability: 0 blocked / 1 passable
2. food: normalized locally visible food amount
3. vegetation: 0/1 forest indicator
4. visibility: 0 masked/unknown / 1 visible (off-world cells masked)

Internal channels: energy, health, hunger, each normalized [0,1]. The agent is always centered; coordinates, world seed, full map and future random draws are **not** observed. Observation encoder must not confuse masked cells with traversable empty ground.

## Transition order

1. Validate action and ensure episode is active.
2. Apply movement/eat/rest attempt.
3. Apply metabolic cost, hunger progression and any environmental effects.
4. Apply deterministic regrowth/random events using environment-local RNG.
5. Determine death and death cause; otherwise mark time limit truncation.
6. Build observation/reward/info/event entry from the resulting state.

Write an exact order in code and tests; avoid off-by-one surprises at step 512. A death on final step takes precedence over time-limit truncation.

## Rewards and termination

First experiment uses a sparse survival objective:

```text
reward = +0.01 on a step survived
reward = -1.0 on the step of death
```

Set `terminated` when health reaches zero (including starvation). Set `truncated` only for the external 512-step cap. Use observed **survival duration**, **food_eaten**, **starvation death rate**, **rest ratio** and **movement diversity** as independent evaluation metrics.

Do not add a food-specific positive reward to the initial experiment unless evidence shows the sparse reward is insufficient; if changed, record it as a versioned experiment/config difference rather than altering results silently.

## World validity invariants

- Agent never occupies an impassable/out-of-bounds tile.
- Resources never become negative or unbounded.
- All observations and rewards are finite.
- Terminal state cannot be stepped without reset.
- Spawn point has viable surroundings under the world-generation constraints.
- Agent cannot obtain energy by repeated free failed actions.
- Resources cannot be duplicated through eat/regrow ordering.

## Test fixtures

Fixed seeds for geometry edges, water bottlenecks, empty-food regions, consumption/regrowth collisions, death on last step, idle strategy exploit, malformed actions and identical trace replay.
