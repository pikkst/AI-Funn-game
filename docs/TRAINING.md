# Training, evaluation and experiment integrity

**Status:** M1 protocol, not completed results.

## Research claim and baselines

Hypothesis: PPO trained from random weights learns better survival strategies than a random action policy. Compare also against simple scripted baselines, e.g. always-rest and eat-when-on-food/random-walk, to rule out trivial reward loopholes.

Do not infer language, self-awareness, intelligence comparable to humans or civilization formation from a good survival score.

## Baseline experiment: Survival-001

| Field | Starting value |
|---|---|
| Environment | survival-v1 |
| Algorithm | PPO (Stable-Baselines3) |
| Network | MlpPolicy, two hidden layers × 128 |
| Training timesteps | 500,000 total across environments |
| Vector environments | 8 CPU environments |
| Episode cap | 512 environment steps |
| Observation | Box(0,1,shape=(199,),dtype=float32) |
| Actions | Discrete(6) |
| Run seeds | at least 3 independent training seeds for final assessment |
| Evaluation | at least 100 held-out episodes per checkpoint/seed |
| Hardware | Windows 11; RTX 2060 SUPER 8 GB; CPU fallback |

Hyperparameters not specified here must be stored in the experiment manifest with explicit values; do not depend silently on changing library defaults. The first smoke training can be 2,000-10,000 timesteps to validate plumbing before the full experiment.

## Split integrity

Predefine and persist disjoint seed sets (`train_world_seeds`, `validation_world_seeds`, `test_world_seeds`). Validation supports checkpoint selection and tuning; report final unbiased results on untouched test seeds. World layout generation must use the world seed, not an accidental coupling to the policy sampling seed.

Reusing a held-out test set repeatedly for hyperparameter selection invalidates its status as an unbiased test.

## Training CLI (planned)

```powershell
python -m ai_funn.training.train --config configs/survival-001.yaml
python -m ai_funn.training.evaluate --run-id <id> --split test
```

**These commands are design targets, not currently runnable.** A training command must write a manifest, metrics, checkpoint(s), and a completion or failure record.

## Metrics and reporting

- mean/median survival steps + distribution and confidence intervals
- death-cause histogram and starvation rate
- food units consumed, food encounters and successful eat actions
- action frequencies, invalid movement, rest share, episode length
- random and scripted baseline side-by-side
- validation vs test performance, across independent training seeds
- training throughput (environment steps/s), wall time, device, peak memory if measurable

Record evaluated checkpoint hash, code revision, environment version and all seed lists. Results without these identifiers are exploratory, not release evidence.

## Acceptance gate for M1

1. Gymnasium environment checker passes.
2. Determinism/regression suite passes on CPU.
3. A fresh policy trains, saves and loads successfully.
4. Frozen-policy evaluation does not update weights.
5. For the registered test protocol, aggregate PPO mean survival is >= 2× random baseline; show per-seed variance and intervals rather than hiding unsuccessful runs.
6. No survival loopholes (always-rest or stationary agent) invalidate reward.
7. Replay produces the same states from serialized initial seed/config/action trace within supported deterministic contract.
8. Document any failure candidly; a failed numerical target triggers investigation, not retroactive threshold reduction.

## Checkpoints and resume

Checkpoint must include or reference environment/action/observation schema version, model architecture, library versions, training step count, seeds and a cryptographic hash. Resuming training should be a deliberate mode and should preserve optimizer state and RNG state to the extent supported. Loading untrusted serialized model files is prohibited.

## Compute practicality

For a small MLP, simulation throughput may be CPU-limited; GPU is not automatically faster due to transfer overhead. Benchmark CPU and CUDA locally before fixing device choice. Limit vector workers for i7-8700K and keep first tests short. No promised fixed training duration.
