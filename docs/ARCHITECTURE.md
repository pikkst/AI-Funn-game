# Technical architecture and contracts

**Status:** planned interfaces; no implementation exists yet.

## System boundaries

```text
                 [Observatory UI]  (M5)
                        |
                   HTTP/WebSocket
                        |
                 [FastAPI adapter] (M5)
                        |
                Read-only snapshots
                        |
[Trainer / Evaluator] --> [Gymnasium SurvivalEnv] --> [Simulation Core]
   PyTorch + PPO           adapter only              authoritative
        |                       |                          |
  checkpoints/          observations/actions         deterministic RNG
  runs/                 reset()/step()               rules + state
```

The simulation core owns truth. The Gymnasium adapter marshals observations, actions and rewards. The learner cannot directly mutate state. UI and API are deferred: there is no reason to introduce an interprocess boundary in M1.

## Target package boundaries

- `simulation/`: state transitions, map generation, resources, organism state, rule configuration, RNG; **no** PyTorch, SB3, FastAPI or PixiJS imports.
- `environments/`: Gymnasium-compatible adapter, observation encoder, action mapping, terminal/truncation handling.
- `agents/`: policy architecture or policy adapters (initially SB3 MlpPolicy).
- `training/`: training CLI, checkpoint policy, logging, resume semantics.
- `experiments/`: validated versioned configuration, seed splits, reproducibility manifests.
- `api/` and `apps/observatory/`: future visualization adapters, never authoritative simulation state.

## Environment API

```python
reset(*, seed: int | None = None, options: dict | None = None) -> tuple[obs, info]
step(action: int) -> tuple[obs, reward: float, terminated: bool, truncated: bool, info: dict]
```

- Gymnasium `Discrete(6)` actions, versioned mapping in [SIMULATION.md](SIMULATION.md).
- Observation is a finite, fixed-shape `float32` vector of 199 values, normalized to [0,1].
- `step` must reject invalid action indices and steps after episode end rather than silently modifying state.
- `reset(seed=s)` seeds the environment-local random generator; never use global random state for world transitions.
- Stable `info` fields should include `step_count`, `food_eaten`, `death_cause` (or null), `world_seed`, and `episode_end_reason` where applicable.
- Terminal death = `terminated=True, truncated=False`; hitting the 512-step external limit = `terminated=False, truncated=True`; final observations remain valid.
- Evaluation uses deterministic actions from a frozen policy. Checkpoint/episode replay uses recorded actions, not a promise of cross-GPU bitwise-deterministic training.

## Determinism contract

Given the same simulator version, rule configuration, world seed and action sequence on the supported CPU numeric path, simulation states and event logs must be identical. Evaluation replay may include serialized initial state and RNG state. Distinguish this from training reproducibility (which can differ across CUDA kernels, parallel scheduling, library versions and hardware).

Tests cover deterministic reset, action-step equivalence, resource placement, invalid actions, world boundaries, terminal step behavior, serialization and seeded regeneration.

## Config and versioning

- All environment/rule changes require an explicit `environment_version` (starting `survival-v1`).
- Action IDs, observation channels, reward semantics and termination criteria are a **versioned contract**.
- Each run manifest includes git commit, Python/library versions, environment version, algorithm, hyperparameters, seeds, device, elapsed time and trained timesteps.
- Incompatible checkpoint/observation/action versions must fail clearly.

## Data storage

M1 uses local files: JSON manifest, TensorBoard events, checkpoint artifacts, evaluation JSON/CSV and replay traces. SQLite is deferred until indexing/search across many runs is useful. Retention policy limits artifact growth; binary checkpoints, runs, caches and secrets are gitignored.

## Security / local execution

Bind future API to loopback by default, require explicit opt-in for LAN binding, protect filesystem paths against traversal, avoid arbitrary code loading from untrusted checkpoints, and do not expose training execution to untrusted network clients.

## Design prohibitions

No hardcoded outcome paths; no hidden full-map observation; no reward computed from future state; no agent writes to core state; no mutation of evaluation policy; no UI dependency in training loop; no duplication of state-transition rules across adapters.
