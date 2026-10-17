# Vision, research questions, and scope

## Goal

Create a small, instrumented artificial-life laboratory where an initially untrained neural organism learns behaviors through interactions with an environment. Subsequent research stages introduce multiple organisms, communication, competition, reproduction, and cultural transmission.

This project intentionally separates **gameplay**, **scientific claims**, and **engineering deliverables**.

## Questions

1. Can an agent learn to consume food and extend survival under local observation?
2. Does that learning generalize to unseen world seeds and resource layouts?
3. Under competition for scarce resources, do distinct policies specialize?
4. Can signaling become useful under cooperative incentives?
5. Which abilities persist across generations, and by which mechanism?

None of these outcomes is presumed. Negative findings are valid experiment results.

## Terms

- **Simulation step:** one transition of world state after an action.
- **Episode:** one lifetime ending in death (terminated) or an external time limit (truncated).
- **Training generation:** not a biological generation; use `training_iteration` to avoid conflation.
- **Biological generation:** descendants produced by an explicitly modeled reproduction mechanism (post-M1).
- **Within-lifetime learning:** updating policy weights from experience, typically between episodes in M1.
- **Evolution:** changes in a population's inherited traits or policies under selection and variation. Not implemented in M1.
- **Emergence:** observed behavior not directly scripted as a goal-specific sequence; requires measurable evidence, appropriate controls, and replication.

## M1 hypothesis

A PPO policy initialized randomly can achieve at least **2.0×** the random-policy mean survival duration across a fixed collection of held-out evaluation world seeds, while consuming food and avoiding death by starvation at a higher rate. This is an **acceptance target**, not an expected outcome or proof of open-ended evolution.

Use multiple training seeds and confidence intervals; avoid depending on a single lucky checkpoint. Acceptance also requires world rules that make indefinite idle survival impossible.

## Product modes (future)

- **Training mode:** headless vectorized environments, checkpoints, metrics, evaluation.
- **Observatory mode:** replay and inspection of world state, actions, outcomes, learning curves; policy decision explanations are limited to available policy inputs and observed outputs, never fabricated reasoning.
- **Arena mode:** fair, frozen-policy head-to-head evaluations.

## Staged scope

- M1: deterministic single-organism survival, random baseline, PPO, held-out evaluation, replay CLI.
- M2: multiple organisms, resource contention, multi-agent environment interface.
- M3: signaling and cooperation studies, control experiments.
- M4: reproduction and inheritance with explicit genetic vs learned-state mechanisms.
- M5: settlements, roles, artifacts, knowledge transmission, richer visualization.

## Risks and mitigations

- **Reward hacking:** measure survival, food consumption, movement and health separately; adversarial edge-case tests.
- **Overfitting:** disjoint train/eval seeds and varied layouts.
- **Non-stationarity:** postpone multi-agent training until M1 has valid single-agent baselines.
- **False emergence claims:** controls against scripted/random policies, repeated trials, independent metrics.
- **Compute limits:** small MLP, 32×32 grid, headless CPU simulation, bounded parallelism.
- **Architecture creep:** do not implement future modes early.
