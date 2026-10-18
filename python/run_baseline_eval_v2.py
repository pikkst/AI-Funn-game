"""DL-013A baseline evaluation under survival-v2: 4 policies x 100 seeds."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ai_funn.agents.baselines import (
    AlwaysRestPolicy,
    EatWhenAvailablePolicy,
    RandomPolicy,
    SimpleWalkerPolicy,
)
from ai_funn.environments.survival_env import SurvivalEnv
from ai_funn.training.evaluation import run_evaluation, save_results

RUNS_DIR = Path(__file__).resolve().parent / "runs"
EVAL_SEEDS = list(range(100))
POLICY_SEED = 1

POLICIES = [
    ("RandomPolicy", RandomPolicy),
    ("AlwaysRestPolicy", AlwaysRestPolicy),
    ("EatWhenAvailablePolicy", EatWhenAvailablePolicy),
    ("SimpleWalkerPolicy", SimpleWalkerPolicy),
]


def main() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    env = SurvivalEnv()
    results = {}
    for name, cls in POLICIES:
        policy = cls(seed=POLICY_SEED)
        result = run_evaluation(env, policy, EVAL_SEEDS, policy_seed=POLICY_SEED)
        results[name] = result
        out_path = RUNS_DIR / f"v2_baseline_{name}_seed100.json"
        save_results(result, out_path)
        print(f"Saved {out_path}")

    print()
    header = (
        f"{'Policy':<26} {'Episodes':>8} {'MeanSurv':>10} {'MedianSurv':>10} "
        f"{'Min':>6} {'Max':>6} {'MeanFood':>9} {'TermRate':>8} {'TruncRate':>8} {'MeanReward':>11}"
    )
    print(header)
    print("-" * len(header))
    for name in list(results.keys()):
        r = results[name]
        death_str = ",".join(f"{k}:{v}" for k, v in sorted(r.death_causes.items()))
        print(
            f"{r.policy_name:<26} {r.num_episodes:>8} {r.mean_survival_steps:>10.4f} "
            f"{r.median_survival_steps:>10.1f} {r.min_survival:>6} {r.max_survival:>6} "
            f"{r.mean_food_consumed:>9.4f} {death_str:<28} "
            f"{r.termination_rate:>8.4f} {r.truncation_rate:>8.4f} {r.mean_reward:>11.4f}"
        )

    print()
    eat = results["EatWhenAvailablePolicy"]
    rest = results["AlwaysRestPolicy"]
    margin = eat.mean_survival_steps - rest.mean_survival_steps
    pct = (margin / rest.mean_survival_steps * 100.0) if rest.mean_survival_steps else 0.0
    print(f"Verification 1 — EatWhenAvailable > AlwaysRest by >=10%: margin={margin:.4f} ({pct:.2f}%) -> {'PASS' if pct >= 10.0 else 'FAIL'}")

    any_trunc = any(r.truncation_rate > 0 for r in results.values())
    print(f"Verification 2 — truncation_rate > 0 for at least one policy: {any_trunc} -> {'PASS' if any_trunc else 'FAIL'}")

    ar = results["AlwaysRestPolicy"]
    ar_ok = ar.termination_rate == 1.0 and ar.truncation_rate == 0.0
    print(f"Verification 3 — AlwaysRest termination_rate=1.0 truncation_rate=0.0: term={ar.termination_rate:.4f} trunc={ar.truncation_rate:.4f} -> {'PASS' if ar_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
