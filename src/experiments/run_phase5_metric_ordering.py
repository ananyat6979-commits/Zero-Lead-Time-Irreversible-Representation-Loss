"""
PHASE 5B:  METRIC ORDERING

Ranks distributional diagnostics by earliest reliable trigger.
"""

import json
import random
from pathlib import Path

from src.experiments.self_training import run_self_training
from src.experiments.config import ExperimentConfig
from src.metrics.distribution import empirical_distribution, js_divergence
from src.metrics.distribution import shannon_entropy, zipf_tail_mass, type_token_ratio

# ------------------
# CONFIG (LOCKED)
# ------------------

INPUT_PATH = Path("results/phase5_diagnostics.json")
OUT_PATH = Path("results/phase5_metric_ordering.json")
TOKEN_PATH = Path("data/processed/pride_and_prejudice.tokens.txt")

K_STD = 2.0  # safe band width (locked)
N_BASELINE_SAMPLES = 10  # matched-generation baseline, replaces the old single-point D0 baseline

METRICS = ["js_to_D0", "entropy", "tail_mass", "ttr"]


def load_tokens(path):
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


# ------------------
# LOAD DATA
# ------------------

with open(INPUT_PATH, "r") as f:
    data = json.load(f)

data = sorted(data, key=lambda x: x["iteration"])

# ------------------
# BASELINE: matched-generation, real variance
# ------------------
# Previously this baseline was a single D0 point with std=0.0 by
# construction, so any nonzero movement at all counted as a trigger.
# Fixed by sampling repeatedly from a model trained on the clean corpus,
# same generation process used elsewhere this session, to get a genuine,
# non-degenerate spread for each metric.

original_tokens = load_tokens(TOKEN_PATH)
original_dist = empirical_distribution(original_tokens)

# Fixed, second attempt: a plain bootstrap resample of original_tokens
# was still a mismatched null model, since the real trajectory in
# results/phase5_diagnostics.json is not an independent resample, it is
# produced by run_phase3.py's actual self-training process at alpha
# equals 0.8, confirmed by reading that file directly this session.
# Fixed properly by running the same process, run_self_training with
# alpha equals 0.8 and iterations equals 1, repeated across several
# seeds, so the baseline represents what a single real self-training
# step looks like under the null hypothesis of no meaningful drift,
# matching the actual generating process of the data being evaluated.

ALPHA = 0.8

baseline_values = {m: [] for m in METRICS}
for seed in range(N_BASELINE_SAMPLES):
    baseline_config = ExperimentConfig(
        model_type="unigram",
        alpha=ALPHA,
        iterations=1,
        sample_size=20,
        random_seed=seed,
    )
    datasets = run_self_training(original_tokens, baseline_config)
    sample = datasets[1]
    sample_dist = empirical_distribution(sample)

    baseline_values["js_to_D0"].append(js_divergence(sample_dist, original_dist))
    baseline_values["entropy"].append(shannon_entropy(sample_dist))
    baseline_values["tail_mass"].append(zipf_tail_mass(sample))
    baseline_values["ttr"].append(type_token_ratio(sample))


def mean_std(xs):
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / len(xs)
    return mu, var ** 0.5


baseline = {}
for m in METRICS:
    mu, sd = mean_std(baseline_values[m])
    baseline[m] = {"mean": mu, "std": sd}


def detect_direction(data, metric):
    # Direction detected empirically from the actual trajectory rather
    # than assumed, since the earlier hardcoded assumptions for entropy,
    # tail_mass, and ttr were verified wrong for this data this session.
    first, last = data[0][metric], data[-1][metric]
    return "increase" if last > first else "decrease"


directions = {m: detect_direction(data, m) for m in METRICS}

# ------------------
# TRIGGER DETECTION
# ------------------

results = []

for metric in METRICS:
    mean = baseline[metric]["mean"]
    std = baseline[metric]["std"]
    direction = directions[metric]

    upper = mean + K_STD * std
    lower = mean - K_STD * std

    trigger_iter = None
    values_after = []

    for row in data:
        val = row[metric]

        crossed = (
            val > upper if direction == "increase"
            else val < lower
        )

        if crossed:
            trigger_iter = row["iteration"]
            values_after = [
                r[metric] for r in data if r["iteration"] >= trigger_iter
            ]
            break

    monotonic = True
    if values_after:
        for a, b in zip(values_after, values_after[1:]):
            if direction == "increase" and b < a:
                monotonic = False
            if direction == "decrease" and b > a:
                monotonic = False

    results.append({
        "metric": metric,
        "trigger_iteration": trigger_iter,
        "direction": direction,
        "monotonic_after": monotonic,
    })

results_sorted = sorted(
    results,
    key=lambda x: float("inf") if x["trigger_iteration"] is None else x["trigger_iteration"]
)

# ------------------
# SAVE
# ------------------

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps({
    "baseline": baseline,
    "ordering": results_sorted
}, indent=2))

print("\nPHASE 5B: METRIC ORDERING\n")
for r in results_sorted:
    print(
        f"{r['metric']:12s} | "
        f"trigger @ {r['trigger_iteration']} | "
        f"direction: {r['direction']} | "
        f"monotonic: {r['monotonic_after']}"
    )

print(f"\nSaved results to {OUT_PATH}")