"""
PHASE 5C — METRIC ORDERING (CANONICAL FIGURE)

Which diagnostic deserves to be trusted for early intervention?

Single figure.
No smoothing.
No collapse narrative.
"""
# NOTE (flagged, not fixed): verified directly that this script currently
# crashes when run against the real results/phase5_metric_ordering.json
# from this session, with TypeError, '>' not supported between instances
# of NoneType and int, at the max(trigger_iters) call. This is downstream
# of two issues already flagged in run_phase5_metric_ordering.py itself,
# the degenerate zero variance baseline and the wrong direction
# assumptions for entropy, tail_mass, and ttr, which together mean only
# js_to_D0 currently has a real trigger_iteration, the other three are
# null. Not resolved here, needs those upstream issues settled first.
import json
from pathlib import Path
import matplotlib.pyplot as plt

# ------------------
# PATHS
# ------------------

DATA_PATH = Path("results/phase5_metric_ordering.json")
OUT_PATH = Path("results/figures/phase5_metric_ordering.png")

# ------------------
# LOAD DATA
# ------------------

with open(DATA_PATH, "r") as f:
    data = json.load(f)

ordering = data["ordering"]

metrics = [m["metric"] for m in ordering]
trigger_iters = [m["trigger_iteration"] for m in ordering]
monotonic = [m["monotonic_after"] for m in ordering]

# ------------------
# STYLE (LOCKED)
# ------------------

plt.style.use("dark_background")

fig, ax = plt.subplots(figsize=(8, 4.5))
fig.patch.set_facecolor("#0B0F19")
ax.set_facecolor("#0B0F19")

ax.grid(axis="x", color="#1F2933", linewidth=0.8, alpha=0.6)
ax.tick_params(colors="#D1D5DB")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color("#374151")
ax.spines["left"].set_color("#374151")

# Modern, restrained palette
COLORS = {
    "js_to_D0": "#60A5FA",   # cool blue
    "entropy": "#C084FC",   # soft violet
    "tail_mass": "#4ADE80", # clean green
    "ttr": "#FACC15",       # warm amber
}

MARKERS = {
    True: "o",   # monotonic
    False: "^",  # non-monotonic
}

# ------------------
# PLOT
# ------------------

y_positions = list(range(len(metrics)))

for i, (metric, x, mono) in enumerate(zip(metrics, trigger_iters, monotonic)):
    ax.scatter(
        x,
        i,
        s=120,
        marker=MARKERS[mono],
        color=COLORS[metric],
        edgecolor="#0B0F19",
        linewidth=1.2,
        zorder=3
    )

# ------------------
# LABELS & ANNOTATION
# ------------------

ax.set_yticks(y_positions)
ax.set_yticklabels(
    [
        "JS divergence (sensitive, unstable)",
        "Entropy (theoretical, noisy)",
        "Tail mass (stable, interpretable)",
        "Type–Token Ratio (stable, coarse)",
    ],
    fontsize=11,
    color="#E5E7EB"
)

ax.set_xlabel("First trigger iteration", color="#D1D5DB")
ax.set_title(
    "Phase 5C — Which Diagnostic Should a System Trust?",
    loc="left",
    fontsize=13,
    color="#E5E7EB"
)

ax.set_xlim(-0.2, max(trigger_iters) + 1)
ax.set_ylim(-0.5, len(metrics) - 0.5)

# Legend (minimal, honest)
ax.text(
    max(trigger_iters) + 0.3,
    len(metrics) - 1.0,
    "● Monotonic after trigger\n▲ Non-monotonic",
    fontsize=10,
    color="#9CA3AF",
    verticalalignment="top"
)

# ------------------
# SAVE
# ------------------

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200)
plt.close()

print(f"Saved figure to {OUT_PATH}")
