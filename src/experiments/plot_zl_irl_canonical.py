# src/experiments/plot_zl_irl_canonical.py

import json
from pathlib import Path
import matplotlib.pyplot as plt

# ---- DATA (loaded live, not hardcoded) ----
# Previously this script hardcoded js_35 and js_36 as literal Python
# lists. Verified this session: those hardcoded values did not match
# a fresh run of run_phase35.py and run_phase36.py against the current
# codebase. js_35 was roughly 47-51% lower than the real values, and
# js_36 was flattened into an instant jump to a flat plateau, when the
# real curve climbs continuously for about 8 iterations before leveling
# off. Fixed by reading both series live from their generating
# scripts' output files instead of frozen numbers.

PHASE35_PATH = Path("results/phase35_diagnostics.json")
PHASE36_PATH = Path("results/phase36_diagnostics.json")

with open(PHASE35_PATH, "r") as f:
    phase35_data = json.load(f)

with open(PHASE36_PATH, "r") as f:
    phase36_data = json.load(f)

iters_35 = [int(d["iteration"].split("=")[1]) for d in phase35_data]
js_35 = [d["js_divergence"] for d in phase35_data]

iters_36 = [d["iteration"] for d in phase36_data]
js_36 = [d["js_divergence"] for d in phase36_data]

threshold = max(js_35)

# ---- STYLE ----
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(10, 5))

# ---- PLOTS ----
ax.plot(iters_35, js_35, color="#9aa0a6", linewidth=2)
ax.plot(iters_36, js_36, color="#ff4d4d", linewidth=3)

ax.axhline(
    threshold,
    linestyle="--",
    color="white",
    linewidth=1.5,
    alpha=0.8,
)

# ---- ANNOTATIONS (DESCRIPTIVE ONLY) ----
# Updated to match what the live data actually shows: a real, gradual
# multi-iteration climb, not an instant jump to a flat plateau.
ax.text(
    0.3,
    threshold * 0.5,
    "BASELINE REGIME",
    color="#9aa0a6",
    fontsize=12,
    va="center",
)

ax.text(
    1.1,
    max(js_36) * 0.55,
    "CONTINUED DIVERGENCE\n(gradual, not instantaneous)",
    color="#ff4d4d",
    fontsize=12,
    va="bottom",
)

ax.text(
    5,
    threshold * 1.05,
    "empirical recovery envelope",
    color="white",
    fontsize=11,
    ha="center",
    alpha=0.85,
)

# ---- AXES ----
ax.set_xlabel("Self-training iteration")
ax.set_ylabel("JS(Pt || P0)")
ax.set_xlim(0, 10)

# ---- FINALIZE ----
plt.tight_layout()
Path("figures").mkdir(parents=True, exist_ok=True)
plt.savefig("figures/zl_irl_canonical.png", dpi=200)
plt.close()

print("Saved figures/zl_irl_canonical.png from live data")