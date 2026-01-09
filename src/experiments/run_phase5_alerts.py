"""
PHASE 5 — EARLY-WARNING ALERT SYSTEM

Consumes Phase 5 distributional diagnostics and assigns
risk states based on statistically defined thresholds.

This does NOT assume collapse.
This does NOT require irreversibility.
This answers: when should we intervene?
"""

import json
from pathlib import Path

from src.diagnostics.thresholds import classify_risk


# ------------------
# CONFIG (LOCKED)
# ------------------

DIAGNOSTICS_PATH = Path("results/phase5_diagnostics.json")
OUT_PATH = Path("results/phase5_alerts.json")

# How many early iterations define the "clean regime"
REFERENCE_ITERATIONS = [0]  # strictly D0


# ------------------
# RISK ASSIGNMENT
# ------------------

def assign_risk(js_value, thresholds):
    """
    Assign discrete risk state based on JS divergence.
    """

    if js_value <= thresholds["safe"]:
        return "SAFE"

    if js_value <= thresholds["high_risk"]:
        return "WARNING"

    return "HIGH_RISK"


# ------------------
# MAIN PIPELINE
# ------------------

def main():
    diagnostics = json.loads(DIAGNOSTICS_PATH.read_text())

    # ---- Reference distribution (clean regime) ----
    reference_js = [
        d["js_to_D0"]
        for d in diagnostics
        if d["iteration"] in REFERENCE_ITERATIONS
    ]

    thresholds = compute_js_thresholds(reference_js)

    alerts = []

    for d in diagnostics:
        risk = assign_risk(d["js_to_D0"], thresholds)

        alerts.append({
            "iteration": d["iteration"],
            "num_tokens": d["num_tokens"],
            "js_to_D0": d["js_to_D0"],
            "entropy": d["entropy"],
            "tail_mass": d["tail_mass"],
            "ttr": d["ttr"],
            "risk_state": risk,
        })

        print(f"[D{d['iteration']}] → {risk}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "thresholds": thresholds,
        "alerts": alerts,
    }, indent=2))

    print(f"\nSaved alert timeline to {OUT_PATH}")


if __name__ == "__main__":
    main()
