# Development roadmap and task tracking

**Status date:** 2026-10-17. This is a planning backlog, **not** a declaration of completed implementation. All checkboxes intentionally start unchecked.

Task IDs are stable within this document. Check an item only when code, tests and the listed acceptance evidence exist.

## Milestone M0 Äā‚¬ā€¯ Documentation and repository foundation

- [x] **DL-001 Äā‚¬ā€¯ Project bootstrap**: add Python package scaffold (`python/pyproject.toml`), lock/pin dependencies, source/test layout and scripts. **AC:** fresh Windows installation can run import + pytest smoke.
- [x] **DL-002 Äā‚¬ā€¯ Quality controls**: Ruff, pytest, type checks where practical, workflow CI, gitignore and documentation links. **AC:** CI executes meaningful validation and excludes large training/checkpoints.
- [x] **DL-003 ā€” Versioned experiment configuration**: typed config, `environment_version`, action/observation versions and seed splits. **AC:** invalid configurations fail before training; manifest serializes exact settings.

## Milestone M1 Äā‚¬ā€¯ First learning organism (priority)

- [x] **DL-010 Äā‚¬ā€¯ Deterministic world core**: seeded 32Ä†ā€”32 world, navigability, resources, regrowth, bounded map generation. **AC:** same seed/config/actions produce identical CPU trace; impossible spawn layouts are rejected.
- [x] **DL-011 Äā‚¬ā€¯ Organism lifecycle**: energy, hunger, health, eat/rest/move, death and time-limit mechanics. **AC:** idle/rest cannot survive indefinitely without food; movement and resource invariants tested.
- [x] **DL-012 Äā‚¬ā€¯ Survival-v1 Gymnasium adapter**: `Discrete(6)`, `float32[199]`, reset/step contract, info, terminal/truncation. **AC:** Gymnasium `check_env` and invalid-action tests pass.
- [x] **DL-013 Äā‚¬ā€¯ Baseline policies**: random, always-rest, eat-on-food and simple walker. **AC:** at least 100 seeded eval episodes with reproducible output and death-cause breakdown.
- [ ] **DL-014 Äā‚¬ā€¯ PPO training CLI**: SB3 MlpPolicy 2Ä†ā€”128, 8 vector environments, short smoke then configurable 500k-step run, bounded artifacts. **AC:** fresh run trains, saves, loads, and logs metrics; CPU works.
- [ ] **DL-015 Äā‚¬ā€¯ Evaluation and scientific integrity**: disjoint world seeds, frozen-policy eval, confidence intervals, per-seed results. **AC:** final protocol shows >=2Ä†ā€” random survival or records an explicit unmet acceptance target without reinterpreting data.
- [ ] **DL-016 Äā‚¬ā€¯ Replay and debugging**: serialize initial config/RNG/seed and action trace, readable event outputs. **AC:** exact deterministic state/event replay on supported CPU path.
- [ ] **DL-017 Äā‚¬ā€¯ End-to-end M1 evidence**: run documented full benchmark and review reward loopholes. **AC:** attach run IDs, manifests, checkpoint hashes, metrics, versions, performance comparison and caveats.

## Milestone M2 Äā‚¬ā€¯ Multiple organisms

- [ ] **DL-020 Äā‚¬ā€¯ Multi-organism world rules**: shared resources, collisions, independent state, action arbitration. **AC:** permutation fairness, no duplicate consumption, deterministic action resolution.
- [ ] **DL-021 Äā‚¬ā€¯ Multi-agent adapter**: PettingZoo-compatible API with identity and turn contract. **AC:** framework validation and stable per-agent reward/accounting.
- [ ] **DL-022 Äā‚¬ā€¯ Competition experiments**: two distinct policies, equal information/action budget, fixed maps. **AC:** repeatable tournaments with baseline controls.

## Milestone M3 Äā‚¬ā€¯ Emergent signaling

- [ ] **DL-030 Äā‚¬ā€¯ Bounded signal channels**: symbolic signaling with measurable cost, no human language assumptions. **AC:** signals have no privileged hidden-state access.
- [ ] **DL-031 Äā‚¬ā€¯ Cooperation benchmarks**: tasks requiring information sharing, plus no-signal/shuffled-signal control groups. **AC:** evaluate communication benefit and generalization.

## Milestone M4 Äā‚¬ā€¯ Inheritance and evolution

- [ ] **DL-040 Äā‚¬ā€¯ Reproduction and selection model**: explicit lifecycle, heredity, variation and population bounds. **AC:** population and inheritance invariants plus reproducible lineage histories.
- [ ] **DL-041 Äā‚¬ā€¯ Separate evolution from training**: define inherited parameters vs learned within-life parameters. **AC:** experiments isolate mechanisms and avoid conflating training steps with biological generations.

## Milestone M5 Äā‚¬ā€¯ Civilization / observatory

- [ ] **DL-050 Äā‚¬ā€¯ Observer API and event protocol**: read-only snapshots/replay + loopback-bound service. **AC:** no direct policy/UI mutation of canonical state.
- [ ] **DL-051 Äā‚¬ā€¯ React/PixiJS visualization**: 2D world, live events, replay and metrics. **AC:** observer doesn't affect deterministic outcomes.
- [ ] **DL-052 Äā‚¬ā€¯ Artifacts, tools and social structures**: gated by demonstrated earlier behaviors. **AC:** explicit mechanics, controls and reproducible experiments.

## Execution order and Definition of Done

Order: DL-001 Äā€ ā€™ DL-002/DL-003 Äā€ ā€™ DL-010 Äā€ ā€™ DL-011 Äā€ ā€™ DL-012 Äā€ ā€™ DL-013 Äā€ ā€™ DL-014 Äā€ ā€™ DL-015 Äā€ ā€™ DL-016 Äā€ ā€™ DL-017. Complete M1 before broadening to M2.

For every task: preserve existing contracts; provide tests, docs, deterministic fixtures where applicable, security/resource considerations, reproducibility notes, and proof that required checks actually ran. No task is complete merely because files were added; no PR is merge-ready if mandatory gates are failing or unknown.

## Blockers and decisions

- Training device (CPU vs CUDA): benchmark after first working env; do not hardcode GPU assumption.
- Hyperparameters and reward tuning: log explicit experimental variants and never tune on untouched final test seeds.
- M1 API/UI: intentionally deferred; avoid premature service architecture.

## Kilo Code task delegation (documentation only)

Use the project-level Kilo agents in `.kilo/agents/`. For DL-001Äā‚¬ā€DL-003 the orchestrator coordinates scaffolding and quality gates within the existing task scope; a coding agent may implement the scaffold, but do not treat narrow specialist edit permissions as permission to bypass the user. DL-010Äā‚¬ā€DL-012 belong to simulation-engineer; DL-014 to rl-engineer; DL-013, DL-015 and DL-017 require research-evaluator plus qa-reviewer. DL-016 should be owned by the implementer of replay with QA verification. M5 observatory work is deferred.

Subagent completion is not task completion: require the actual diff, tests, reproducibility metadata, relevant acceptance evidence and user-approved integration. See [docs/KILO_CODE.md](docs/KILO_CODE.md). No tasks are marked complete by the documentation integration.


