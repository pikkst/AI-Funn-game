# DL-013A Calibration Plan

**Status:** Approved — implemented
**Owner:** simulation-engineer  
**Related:** DL-013 (baseline evaluation), DL-014 (PPO training), ARCHITECTURE.md

---

## 1. Problem statement

DL-013 baselines show `AlwaysRestPolicy` (308.0 steps mean survival) outperforms all active food-seeking policies:

| Policy | Mean survival (steps) |
|---|---:|
| AlwaysRestPolicy | 308.0 |
| EatWhenAvailablePolicy | 224.05 |
| RandomPolicy | 226.06 |
| SimpleWalkerPolicy | 199.0 |

All 400 episodes terminated by starvation; none reached the 512-step truncation cap.  
The current reward (+0.01/step, −1.0 on death) incentivizes passive rest-forever behavior rather than food seeking.

---

## 2. Root cause analysis

### 2.1 Hunger reaches max in 10 steps

Hunger increments at `0.1` per step:

```
steps to hunger 1.0 = 1.0 / 0.1 = 10 steps
```

An agent that rests indefinitely without food starves after 10 steps unless metabolism is offset by rest recovery.

### 2.2 AlwaysRest net energy

Rest recovery: `2.0 × (1 − hunger)`, metabolism: `−0.5`.

Net energy per rest step:

```
net_rest(hunger) = −0.5 + 2.0 × (1 − hunger)
```

Rest is energy-positive while:

```
−0.5 + 2.0 × (1 − hunger) > 0
2.0 × (1 − hunger) > 0.5
1 − hunger > 0.25
hunger < 0.75
```

So AlwaysRest accumulates net-positive energy until hunger exceeds 0.75, then drifts toward starvation slowly. This is why it survives ~308 steps while active policies die faster.

### 2.3 Movement vs eating is net-negative (corrected)

- Movement cost: `1.0` per step (0.5 metabolism + 0.5 movement extra)
- **Eat step cost: 0.5 metabolism ONLY** — action 4 (EAT) is not in (0,1,2,3), so `MOVEMENT_EXTRA_COST` is not applied (`world.py:248-258`)
- Eating one tile restores: `1.0` food

Net after eating:

```
net_eat = −0.5 (metabolism) + 1.0 (food restored) = +0.5
```

Reaching food costs ≥1.0 movement before the eat step, so:

```
foraging net = +0.5 (eat) − ≥1.0 (movement to reach food) = ≤ −0.5
```

Food is only worth pursuing when adjacent (zero travel cost), where eat alone nets +0.5. With hunger incrementing 0.1/step, an agent burns more energy traveling to food than it gains from eating.

### 2.4 Death penalty equals 100 steps of survival reward

Survival reward: `+0.01` per step. Death penalty: `−1.0`.

```
steps of survival reward equal to death penalty = 1.0 / 0.01 = 100 steps
```

Dying after 300 steps yields `300 × 0.01 − 1.0 = 2.0` total reward. Surviving 100 steps then dying yields `100 × 0.01 − 1.0 = 0.0`. There is no urgency to avoid death: an agent earns back the full death penalty with just 100 more steps of survival, so risk-averse resting dominates risky foraging.

---

## 3. Parameter sweep results

100 episodes per config, seeds 0–99, `policy_seed=1`. All runs properly monkeypatched (food grid init, regrow parameters, metabolism constants).

| Config | Changes | Random | AlwaysRest | EatWhenAvailable | SimpleWalker | EWA > AR? | Truncation? |
|---|---|---:|---:|---:|---:|---|---|
| A — baseline | none | 226.1 | 308.0 | 224.1 | 199.0 | No | No |
| B — 3× food | food 1.0 → 3.0 | 239.0 | 308.0 | 280.4 | 199.0 | No | No |
| C — cheaper move | move cost 1.0 → 0.7 | 267.0 | 308.0 | 277.6 | 242.0 | No | No |
| D — slower hunger | hunger 0.1 → 0.05 | 227.8 | 316.0 | 224.1 | 199.0 | No | No |
| E — combined (rest 2.0) | food 3.0, move 0.7, hunger 0.05 | 292.1 | 316.0 | 393.9 | 242.0 | Yes | 4% |
| F — full v2 (rest 1.5) | food 3.0, move 0.7, hunger 0.05, rest 1.5 | 291.2 | 316.0 | 393.9 | 242.0 | Yes | 4% |

Config E and F both show `EatWhenAvailablePolicy` exceeding `AlwaysRestPolicy` with truncations occurring. Rest recovery 1.5 vs 2.0 makes no difference to AlwaysRest (hunger reaches max before recovery differential matters), confirming rest_recovery is not a critical parameter.

---

## 4. Recommended calibration (Config F)

Bump environment version `survival-v1` → `survival-v2`.

| Parameter | v1 (current) | v2 (recommended) | Location |
|---|---:|---:|---|
| Food amount per tile | 1.0 | 3.0 | `init_food_grid_from_terrain`, `regrow_resources` |
| Movement extra cost | 0.5 | 0.2 (total move 1.0 → 0.7) | `world.py` step logic |
| Hunger increment | 0.1 | 0.05 | `world.py` metabolism |
| Rest recovery multiplier | 2.0 | 1.5 | `world.py` rest logic |
| Survival reward | +0.01/step | +0.01/step (unchanged) | `survival_env.py` |
| Death penalty | −1.0 | −1.0 (unchanged) | `world.py` terminal |

### Predicted outcomes under v2 (observed in Config F)

- `EatWhenAvailablePolicy` mean survival > `AlwaysRestPolicy` by 24.6% margin (393.9 vs 316.0)
- 4% of episodes reach the 512-step truncation cap (4/100)
- `RandomPolicy` (291.2) below `EatWhenAvailablePolicy` (393.9) — food-seeking outperforms random
- All deaths remain starvation; truncation is the only non-starvation outcome

---

## 5. Implementation requirements

1. **No production code changes until this plan is approved.**
2. **Environment version bump required** (`survival-v1` → `survival-v2`), per ARCHITECTURE.md:57.
3. **Version-bump code-change checklist:**
   - `python/src/ai_funn/experiments/config.py:21` — `ENVIRONMENT_VERSION = "survival-v1"` → `"survival-v2"`
   - `python/src/ai_funn/experiments/config.py:58-62` — `EnvironmentConfig.__post_init__` version validation must accept `"survival-v2"`
   - `python/src/ai_funn/simulation/world.py:162-163` — version validation must accept `"survival-v2"`
   - `python/src/ai_funn/environments/survival_env.py:34` — `ENVIRONMENT_VERSION = "survival-v1"` → `"survival-v2"`
   - `docs/SIMULATION.md` — title and references to "survival-v1" → "survival-v2"
   - `python/tests/test_config.py` — update any hardcoded `"survival-v1"` assertions
4. **All DL-010 / DL-011 / DL-012 regression tests must pass under v2.**
5. **New tests with specific acceptance criteria:**
   - `test_food_seeking_beats_rest`: EatWhenAvailablePolicy mean survival > AlwaysRestPolicy by ≥10% under v2
   - `test_truncation_achievable`: ≥1 of 100 episodes reaches truncation under v2
   - `test_no_idle_exploit`: AlwaysRestPolicy does not exceed EatWhenAvailablePolicy under v2
   - `test_baseline_replay_determinism`: same seeds → identical results under v2
6. **Baseline results JSON saved under v2** for DL-014 PPO comparison.

---

## 6. Decision needed

Approve or modify the calibration targets and parameters listed in Section 4. Once approved, implementation proceeds under DL-013A with version bump to `survival-v2` and validation per Section 5.
