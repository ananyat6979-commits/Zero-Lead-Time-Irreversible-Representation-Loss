import json
from pathlib import Path

import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------

# SAFE_THRESHOLD was previously hardcoded at 0.009794292127964984, derived
# from a pre mixture-fix run of Phase 3.5 (see e60ed63 and da49ef2 in git
# history). That value no longer matches results/phase35_diagnostics.json
# once regenerated under the corrected pipeline, real js_divergence values
# there are around 0.0143 to 0.0148, not around 0.0098. Recomputing here
# directly from the same file this script already loads, so the threshold
# and the data being compared to it always come from the same run.


# NOTE (flagged, not fully resolved): recomputing SAFE_THRESHOLD as
# max(js_divergence) from the same phase35 data plotted in Panel A means
# the threshold is definitionally anchored to Panel A's own maximum value.
# The dashed line will always sit at or just above Panel A's highest point
# by construction, which is a much weaker claim than "Panel A stays safely
# below an independently derived threshold." This is better than comparing
# fresh data against a stale, pre-fix constant, but it does not fully
# restore the panel's original rhetorical claim. A genuinely independent
# safe threshold would need to come from a source other than the same
# data being visually judged against it. Not resolved here.


phase35_for_threshold = json.loads(
    Path("results/phase35_diagnostics.json").read_text()
)
SAFE_THRESHOLD = max(d["js_divergence"] for d in phase35_for_threshold)
OUTPUT_PATH = Path("results/zl_irl_exhibit.png")

# Modern dark palette (hand-picked, no defaults)
COLORS = {
    "background": "#0e1117",
    "panel": "#161b22",
    "text": "#e6edf3",
    "muted": "#8b949e",
    "safe": "#2f81f7",
    "failure": "#f78166",
    "threshold": "#d29922",
}

# -----------------------------
# Load diagnostics
# -----------------------------

phase35 = json.loads(
    Path("results/phase35_diagnostics.json").read_text()
)
phase36 = json.loads(
    Path("results/phase36_diagnostics.json").read_text()
)

# -----------------------------
# Prepare data
# -----------------------------

k_values = [int(d["iteration"].split("=")[1]) for d in phase35]
js_phase35 = [d["js_divergence"] for d in phase35]

iters = [d["iteration"] for d in phase36]
js_phase36 = [d["js_divergence"] for d in phase36]

# -----------------------------
# Plot
# -----------------------------

plt.style.use("default")
fig, axes = plt.subplots(
    1, 2,
    figsize=(14, 5),
    sharey=True,
    facecolor=COLORS["background"]
)

for ax in axes:
    ax.set_facecolor(COLORS["panel"])
    ax.tick_params(colors=COLORS["text"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["muted"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.xaxis.label.set_color(COLORS["text"])
    ax.title.set_color(COLORS["text"])

# ---- Panel A: Negative Control ----

axes[0].plot(
    k_values,
    js_phase35,
    marker="o",
    linewidth=2.5,
    color=COLORS["safe"],
    label="JS divergence"
)

axes[0].axhline(
    SAFE_THRESHOLD,
    linestyle="--",
    linewidth=1.5,
    color=COLORS["threshold"],
    label="SAFE threshold"
)

axes[0].set_title("A. Reversible Regime (Negative Control)")
axes[0].set_xlabel("Contamination depth (k)")
axes[0].set_ylabel("JS divergence to D₀")

axes[0].legend(
    frameon=False,
    labelcolor=COLORS["text"]
)

# ---- Panel B: Constraint Introduced ----

axes[1].plot(
    iters,
    js_phase36,
    marker="o",
    linewidth=2.5,
    color=COLORS["failure"],
    label="JS divergence"
)

axes[1].axhline(
    SAFE_THRESHOLD,
    linestyle="--",
    linewidth=1.5,
    color=COLORS["threshold"],
    label="SAFE threshold"
)

axes[1].set_title("B. Constraint Introduced")
axes[1].set_xlabel("Self-training iteration")

axes[1].legend(
    frameon=False,
    labelcolor=COLORS["text"]
)

# -----------------------------
# Final layout
# -----------------------------

fig.suptitle(
    "Zero-Lead-Time Irreversible Representation Loss (ZL-IRL)",
    color=COLORS["text"],
    fontsize=14,
    y=1.03
)

plt.tight_layout()
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    facecolor=COLORS["background"],
    bbox_inches="tight"
)
plt.close()

print(f"Saved exhibit to {OUTPUT_PATH}")
