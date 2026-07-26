"""
PHASE 5B — METRIC ORDERING

Ranks distributional diagnostics by earliest reliable trigger.
No new data. No threshold tuning. No visualization.
"""

import json
from pathlib import Path
import statistics

# ------------------
# CONFIG (LOCKED)
# ------------------

INPUT_PATH = Path("results/phase5_diagnostics.json")
OUT_PATH = Path("results/phase5_metric_ordering.json")

K_STD = 2.0  # safe band width (locked)

# NOTE (flagged, not fixed): two structural issues found when this script
# was first run against real data this session.
#
# 1. js_to_D0 baseline mean is always exactly 0.0 (D0 compared to itself),
#    and std is 0.0 by design (single point baseline). This means upper
#    and lower both equal 0.0 exactly, so any nonzero divergence at all
#    counts as crossed. The observed trigger_iteration of 1 for js_to_D0
#    is therefore guaranteed by construction, not a genuine early signal
#    relative to the other metrics.
#
# 2. The direction assumed below for entropy, tail_mass, and ttr does not
#    match what was actually observed in results/phase5_diagnostics.json
#    this session. entropy rose instead of decreasing, tail_mass fell
#    instead of increasing, and ttr fell instead of increasing, across
#    D1 through D10. Since each metric here only checks its assumed
#    direction, all three never trigger, shown as trigger_iteration null,
#    even though they clearly moved away from baseline, just in the
#    opposite direction from what is assumed. This may reflect a genuine
#    property of this corpus and contamination setup, or a wrong
#    assumption baked into METRICS below. Not resolved here, needs a
#    modeling decision, not a code fix.
METRICS = {
    "js_to_D0": {"direction": "increase"},
    "entropy": {"direction": "decrease"},
    "tail_mass": {"direction": "increase"},
    "ttr": {"direction": "increase"},
}


# ------------------
# LOAD DATA
# ------------------

with open(INPUT_PATH, "r") as f:
    data = json.load(f)

# Index by iteration
data = sorted(data, key=lambda x: x["iteration"])

# ------------------
# BASELINE (D0 ONLY)
# ------------------

baseline = {}
d0 = data[0]

for m in METRICS:
    baseline[m] = {
        "mean": d0[m],
        "std": 0.0  # single point baseline (explicit, honest)
    }

# ------------------
# TRIGGER DETECTION
# ------------------

results = []

for metric, cfg in METRICS.items():
    mean = baseline[metric]["mean"]
    std = baseline[metric]["std"]
    direction = cfg["direction"]

    upper = mean + K_STD * std
    lower = mean - K_STD * std

    trigger_iter = None
    values_after = []

    for row in data[1:]:
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

    # Monotonicity check
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
        "monotonic_after": monotonic
    })

# Sort by earliest trigger
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

# ------------------
# PRINT SUMMARY
# ------------------

print("\nPHASE 5B — METRIC ORDERING\n")
for r in results_sorted:
    print(
        f"{r['metric']:12s} | "
        f"trigger @ {r['trigger_iteration']} | "
        f"monotonic: {r['monotonic_after']}"
    )

print(f"\nSaved results to {OUT_PATH}")

