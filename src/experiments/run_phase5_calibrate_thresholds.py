"""
PHASE 5.26: DELTA-BASED THRESHOLD CALIBRATION (LOCKED)

Calibrates WARNING / HIGH_RISK thresholds on
Δ(metric) relative to the clean reference distribution (D0).
"""

import json
from pathlib import Path

from src.metrics.distribution import zipf_tail_mass
from src.experiments.self_training import run_self_training
from src.experiments.config import ExperimentConfig

DATA_PATH = Path("data/processed/pride_and_prejudice.tokens.txt")
OUT_PATH = Path("results/phase5_thresholds.json")

N_BOOT = 50
K_WARNING = 1.0
K_HIGH = 3.0

# Verified this session: the previous plain bootstrap resample of the
# full corpus produced a mean delta of about -0.002535, but the actual
# ALPHA used by run_phase5_warmstart_alerts.py and
# run_phase6_iteration_boundary.py is 0.1, and a properly alpha matched
# one step self-training baseline at that alpha produces a mean delta of
# about -0.001163, a ratio of about 2.18x, not a small difference. This
# is the same shape of mismatch found and fixed three times in
# run_phase5_metric_ordering.py earlier this session. Fixed the same
# way, by using the real alpha matched generating process instead of an
# unrelated full resample.
#
# Fixed a second, separate issue: the threshold formula used
# mu + K*sd for both warning and high_risk. Since mu is negative here
# (tail mass shrinks under contamination) and sd is positive, adding a
# larger K moved the threshold toward zero, toward SAFE, not further
# into the collapse tail. This made high_risk numerically less extreme
# than warning, the opposite of what K_HIGH=3.0 > K_WARNING=1.0 was
# meant to express. classify_risk's sorted-magnitude comparison masked
# this by making the bands valid regardless of which one is larger, but
# the calibration itself was computing the wrong pair of numbers. Fixed
# by subtracting: mu - K*sd, so high_risk is now genuinely further into
# the negative tail than warning, by construction, without relying on
# classify_risk's sort to paper over it.
ALPHA = 0.1


def load_tokens(p):
    return p.read_text(encoding="utf-8", errors="replace").splitlines()


def mean_std(xs):
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / len(xs)
    return mu, var ** 0.5


def main():
    tokens = load_tokens(DATA_PATH)
    base_tail = zipf_tail_mass(tokens)

    deltas = []
    for seed in range(N_BOOT):
        config = ExperimentConfig(
            model_type="unigram",
            alpha=ALPHA,
            iterations=1,
            sample_size=20,
            random_seed=seed,
        )
        datasets = run_self_training(tokens, config)
        sample = datasets[1]
        deltas.append(zipf_tail_mass(sample) - base_tail)

    mu, sd = mean_std(deltas)

    thresholds = {
        "tail_mass_delta": {
            "warning": mu - K_WARNING * sd,
            "high_risk": mu - K_HIGH * sd,
        }
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(thresholds, indent=2))
    print(f"Saved calibrated delta thresholds to {OUT_PATH}")


if __name__ == "__main__":
    main()