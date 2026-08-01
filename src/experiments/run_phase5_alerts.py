"""
PHASE 5 — EARLY-WARNING ALERTING (DELTA-BASED)
"""

import json
from pathlib import Path
from src.diagnostics.thresholds import classify_risk

DIAGNOSTICS_PATH = Path("results/phase5_diagnostics.json")
OUT_PATH = Path("results/phase5_alerts.json")


def main():
    diagnostics = json.loads(DIAGNOSTICS_PATH.read_text())

    base_tail = diagnostics[0]["tail_mass"]

    alerts = []
    for d in diagnostics:
        delta = d["tail_mass"] - base_tail
        risk = classify_risk("tail_mass_delta", delta)

        alerts.append({
            "iteration": d["iteration"],
            "tail_mass": d["tail_mass"],
            "delta_tail_mass": delta,
            "risk_state": risk,
        })

        print(f"[D{d['iteration']}] delta_tail={delta:.6f} -> {risk}")
        

    OUT_PATH.write_text(json.dumps({
        "baseline_tail_mass": base_tail,
        "alerts": alerts
    }, indent=2))

    print(f"Saved Phase 5 alerts to {OUT_PATH}")


if __name__ == "__main__":
    main()
