# Instructions for AI coding agents — AI-Funn-game

Read `README.md`, `TASKS.md` and relevant files in `docs/` before touching code. This file applies repository-wide unless a more specific `AGENTS.md` is introduced later.

## Mission and operating mode

Build a **local-first artificial-life experiment**, not a chat-bot roleplay of a civilization. The primary scientific deliverable is measured learning of a randomly initialized neural policy in a rules-driven environment. Treat proposed features and results as hypotheses until implemented and measured.

The default task is **analysis or implementation only within the user's requested scope**. Do not commit, push, merge, deploy, close tasks, or create external resources without explicit authorization for that action. Do not expand a narrowly requested fix into a new subsystem.

## Canonical documents

- `docs/VISION.md`: hypothesis and scope.
- `docs/ARCHITECTURE.md`: module boundaries, data contracts, determinism.
- `docs/SIMULATION.md`: survival-v1 world/actions/observations/rewards.
- `docs/TRAINING.md`: experimental integrity, metrics, evaluation.
- `docs/DEVELOPMENT.md`: environment, CI and local commands.
- `TASKS.md`: priorities, acceptance criteria and completion tracking.

When documents conflict, report and reconcile the conflict before implementing. Do not silently break a versioned environment contract.

## Coding rules

1. **Simulation is authoritative.** Agents output actions only; no direct writes to world state.
2. **Separate concerns.** Pure/minimally stateful simulation core must not import ML, UI, API or storage modules.
3. **Deterministic CPU transitions.** Use environment-local seeded RNG; no ambient global randomness in state transitions. Record config/seed/action sequences for replay.
4. **Stable Gymnasium contract.** Observation `float32[199]`, action `Discrete(6)`, valid reset/step semantics, explicit death vs truncation. Schema changes need an environment-version bump and matching docs/tests.
5. **No hidden information.** Policy sees only documented local observations, not full map, world seed, random state, future events or privileged info dictionary.
6. **No reward shortcuts.** Check that idle/rest/invalid-action loops cannot exploit metabolism or reward; benchmark against scripted baselines.
7. **No test gaming.** Fix the implementation when it violates documented behavior; change tests only when the actual contract was wrong, explaining why.
8. **No invented evidence.** Never label a benchmark, CI run, device test, or GPU training 'passed' unless it actually ran.
9. **No premature abstractions.** Prefer clear small modules and existing dependencies; avoid frameworks/services before necessary.
10. **No unbounded work.** Bound episodes, retries, workers, resources, artifacts and checkpoint retention.
11. **Cross-platform and types.** Primary target Windows 11/PowerShell; be explicit about Python and dependency versions. Comments and code identifiers in English.
12. **Security.** No secrets, unsafe untrusted checkpoint loading, arbitrary code execution from config, insecure network exposure or unsanitized file paths.

## Required workflow

1. Read actual current repo code and the relevant task AC; identify dependencies and scope.
2. State a concise implementation plan for nontrivial changes.
3. Make smallest coherent change without duplicating existing code or rewriting unrelated areas.
4. Add unit/integration/property tests for invariants and edge cases: spawn validity, boundaries, resource conservation, terminal handling, seed reproducibility, agent information boundaries, reward exploit loops.
5. Run focused tests, lint/type checks and available CI equivalent; report commands and actual results.
6. Update contracts/documentation and relevant `TASKS.md` boxes only when supported by implementation + evidence.
7. For PR review: read actual diff, AC, tests and CI; report prioritized findings with file/line, risk and fix. Review-only must not change code.

## Scientific evaluation requirements

- Separate training, validation, final test worlds/seeds.
- Never modify a test set for hyperparameter selection.
- Freeze weights during evaluation; use deterministic inference where configured.
- Report survival distributions, food consumption, starvation death, action statistics and confidence intervals; benchmark random, always-rest and simple heuristic policies.
- Record git revision, environment/action/observation versions, dependency versions, seed sets, training timesteps, device, checkpoint identity.
- Do not conflate within-lifetime RL with evolutionary inheritance.
- A 2× random-baseline result is a target, **not** an assumed fact.

## Definition of Done

Task AC met; tests and appropriate gates actually passed; reproducible artifacts recorded; docs updated; no unauthorized writes/merge/deploy; known limitations reported. If unknown or blocked, state precisely what remains and do not mark done.
