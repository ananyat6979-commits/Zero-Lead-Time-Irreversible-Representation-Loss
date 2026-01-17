# src/experiments/plot_zl_irl_canonical.py

import matplotlib.pyplot as plt

# ---- DATA (DO NOT RECOMPUTE) ----
iters_35 = [3, 5, 7, 10]
js_35 = [
    0.009789413940857675,
    0.00978946327744742,
    0.009794292127964984,
    0.009792349400900045,
]

iters_36 = list(range(11))
js_36 = [
    0.0,
    0.036173830966108295,
    0.03617593220535612,
    0.03617177895563627,
    0.03617720342596831,
    0.036170318252980775,
    0.036176902836393174,
    0.03617768067013228,
    0.03617083083470686,
    0.036168339691027424,
    0.0361795360864068,
]

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
    max(js_36) * 1.02,
    "POST-ENTRY REGIME",
    color="#ff4d4d",
    fontsize=13,
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
ax.set_ylabel("JS(Pₜ ∥ P₀)")
ax.set_xlim(0, 10)

# ---- FINALIZE ----
plt.tight_layout()
plt.savefig("figures/zl_irl_canonical.png", dpi=200)
plt.close()
