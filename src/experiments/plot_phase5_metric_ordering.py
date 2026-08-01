"""
PHASE 5C: METRIC ORDERING (CANONICAL FIGURE)

Which diagnostic deserves to be trusted for early intervention?

Single figure.
No smoothing.
No collapse narrative.
"""
# NOTE (fixed): the two upstream issues this comment used to describe
# (degenerate zero-variance baseline; hardcoded direction assumptions)
# are already fixed in the current run_phase5_metric_ordering.py, it
# now uses a matched-generation baseline with real variance and detects
# each metric's direction empirically. Re-run fresh: js_to_D0, tail_mass,
# and ttr all genuinely trigger at iteration 2; entropy genuinely never
# crosses its band across the full trajectory (oscillates within ~0.008
# of its own baseline mean the entire run), that's a real finding
# (entropy is a weak discriminator at this contamination level, alpha
# equals 0.8), not a bug. The crash below was purely about max() choking
# on that legitimate None, not about the underlying data being wrong.
#
# Fixed two things: (1) the axis-limit max() now ignores None entries
# instead of crashing on them, and entropy is plotted as an explicit
# open marker past the right edge with a "never triggered" label rather
# than being silently dropped; (2) y-axis labels are now built directly
# from `ordering`'s own metric order instead of a hardcoded list, since
# a hardcoded label list silently mismatches if `ordering`'s sort order
# ever changes (it's sorted by trigger_iteration, which is data-dependent).
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

fig, ax = plt.subplots(figsize=(8, 5.2))
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

# Human-readable descriptors, keyed by metric name so they follow
# `ordering`'s actual sort order rather than assuming a fixed position.
DESCRIPTIONS = {
    "js_to_D0": "JS divergence",
    "entropy": "Entropy",
    "tail_mass": "Tail mass",
    "ttr": "Type\u2013Token Ratio",
}

# ------------------
# NEVER-TRIGGERED HANDLING
# ------------------
# entropy genuinely never crosses its band in the current data -- a real
# finding, not missing data. Plot it at a fixed sentinel x-position past
# the real triggers, clearly marked, instead of crashing on None or
# silently dropping the row.

real_triggers = [x for x in trigger_iters if x is not None]
x_max_real = max(real_triggers) if real_triggers else 0
NEVER_TRIGGERED_X = x_max_real + 1.5

plot_x = [x if x is not None else NEVER_TRIGGERED_X for x in trigger_iters]

# ------------------
# PLOT
# ------------------

y_positions = list(range(len(metrics)))

for i, (metric, x, x_orig, mono) in enumerate(zip(metrics, plot_x, trigger_iters, monotonic)):
    ax.scatter(
        x,
        i,
        s=120,
        marker=MARKERS[mono] if x_orig is not None else "x",
        color=COLORS[metric],
        edgecolor="#0B0F19",
        linewidth=1.2,
        zorder=3
    )
    if x_orig is None:
        ax.text(
            x + 0.15, i, "never triggered",
            fontsize=9, color="#9CA3AF", verticalalignment="center"
        )

# ------------------
# LABELS & ANNOTATION
# ------------------

ax.set_yticks(y_positions)
ax.set_yticklabels(
    [DESCRIPTIONS.get(m, m) for m in metrics],
    fontsize=11,
    color="#E5E7EB"
)

ax.set_xlabel("First trigger iteration", color="#D1D5DB")
ax.set_title(
    "Phase 5C: Which Diagnostic Should a System Trust?",
    loc="left",
    fontsize=13,
    color="#E5E7EB"
)

ax.set_xlim(-0.2, NEVER_TRIGGERED_X + 2.2)
ax.set_ylim(-0.5, len(metrics) - 0.5)

# Legend, fixed properly this time: reserve real space for it using
# subplots_adjust, rather than placing text at a raw figure coordinate
# that tight_layout does not know to leave room for, which clipped the
# legend at the bottom edge of the saved image in the previous attempt.
plt.subplots_adjust(bottom=0.22)
fig.text(
    0.5,
    0.06,
    "\u25cf Monotonic after trigger    \u25b2 Non-monotonic    \u2715 Never triggered",
    fontsize=10,
    color="#9CA3AF",
    ha="center",
)

# ------------------
# SAVE
# ------------------

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=200)
plt.close()

print(f"Saved figure to {OUT_PATH}")