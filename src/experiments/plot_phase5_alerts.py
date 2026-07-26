"""
PHASE 5: EARLY-WARNING ALERT TIMELINE (CANONICAL FIGURE)

Single visualization.
No smoothing.
No collapse narrative.
"""
# NOTE (flagged, not fixed): this script expects results/phase5_alerts.json
# in the shape produced by run_phase5_early_warning.py, a dict with
# thresholds and alerts keys, where each alert record has iteration,
# js_to_D0, and risk_state. The file currently on disk this session is a
# flat list of records shaped differently, iteration, risk_state, and a
# nested metrics dict with js_divergence and tail_mass, not js_to_D0 at
# the top level. Running this script against that file raises
# TypeError: list indices must be integers or slices, not str, since
# raw is a list, not a dict, in that version.
# This is the same schema conflict already flagged in
# run_phase5_early_warning.py. Not resolved here.

import json
from pathlib import Path
import matplotlib.pyplot as plt

# ------------------
# PATHS
# ------------------

ALERTS_PATH = Path("results/phase5_alerts.json")
OUT_PATH = Path("results/figures/phase5_alert_timeline.png")

# ------------------
# LOAD DATA (CANONICAL PHASE 5 SCHEMA)
# ------------------

with open(ALERTS_PATH, "r") as f:
    raw = json.load(f)

thresholds = raw["thresholds"]
records = raw["alerts"]

iterations = [r["iteration"] for r in records]
js_values = [r["js_to_D0"] for r in records]
alerts = [r["risk_state"] for r in records]

if not iterations:
    raise RuntimeError("Phase 5 alerts contain no records")


# ------------------
# STYLE (LOCKED)
# ------------------

plt.style.use("dark_background")

fig, ax = plt.subplots(figsize=(10, 4.5))
fig.patch.set_facecolor("#0B0F19")
ax.set_facecolor("#0B0F19")

ax.grid(True, color="#1F2933", linewidth=0.8, alpha=0.6)
ax.tick_params(colors="#D1D5DB")
ax.spines["bottom"].set_color("#374151")
ax.spines["left"].set_color("#374151")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Modern, tasteful palette (NOT red/amber cliché)
COLORS = {
    "SAFE": "#4ADE80",       # soft green
    "WARNING": "#FACC15",    # warm yellow
    "HIGH_RISK": "#F472B6",  # restrained pink
}

BANDS = {
    "SAFE": "#0F172A",
    "WARNING": "#2E2A1F",
    "HIGH_RISK": "#3A1F2A",
}

# ------------------
# PLOT
# ------------------

# Background regime bands (semantic, not numeric)
for i in range(len(iterations) - 1):
    ax.axvspan(
        iterations[i],
        iterations[i + 1],
        color=BANDS[alerts[i]],
        alpha=0.6,
        zorder=0
    )

# Main JS trajectory
ax.plot(
    iterations,
    js_values,
    color="#E5E7EB",
    linewidth=1.6,
    zorder=2
)

# Alert-colored points
for x, y, a in zip(iterations, js_values, alerts):
    ax.scatter(x, y, s=70, color=COLORS[a], zorder=3)

# ------------------
# FIRST HIGH_RISK MARKER (ONLY IF IT EXISTS)
# ------------------

high_risk_iters = [
    i for i, a in zip(iterations, alerts) if a == "HIGH_RISK"
]

if high_risk_iters:
    first_high_risk = high_risk_iters[0]

    ax.axvline(
        first_high_risk,
        linestyle=":",
        linewidth=1.5,
        color="#9CA3AF",
        alpha=0.9,
        zorder=1
    )

    ax.text(
        first_high_risk + 0.1,
        max(js_values) * 0.96,
        "First HIGH_RISK",
        color="#9CA3AF",
        fontsize=10,
        va="top"
    )


# ------------------
# LABELS
# ------------------

ax.set_xlabel("Self-training iteration", color="#D1D5DB")
ax.set_ylabel("JS divergence to original (D0)", color="#D1D5DB")
ax.set_title(
    "Phase 5 — Early-Warning Diagnostic Timeline",
    loc="left",
    color="#E5E7EB"
)

ax.set_xlim(min(iterations) - 0.5, max(iterations) + 0.5)
ax.set_ylim(0, max(js_values) * 1.1)

# ------------------
# SAVE
# ------------------

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200)
plt.close()

print(f"Saved figure to {OUT_PATH}")
