import json
from pathlib import Path

# NOTE (flagged, not fixed): this script writes results/phase5_thresholds.json
# using a DIFFERENT method and schema than run_phase5_calibrate_thresholds.py:
# this one derives {"thresholds": {"safe": ..., "high_risk": ...}} from
# max(phase3.5's raw js_divergence) * [1, 2], while
# run_phase5_calibrate_thresholds.py derives {"tail_mass_delta": {"warning":
# ..., "high_risk": ...}} via bootstrap resampling. These are NOT
# interchangeable, run_phase5_warmstart_alerts.py and
# run_phase6_iteration_boundary.py both require the tail_mass_delta schema.
# Running this script after run_phase5_calibrate_thresholds.py will silently
# overwrite results/phase5_thresholds.json back into the incompatible,
# stale schema and break both of those scripts again. DO NOT RUN this
# script until this conflict is resolved, committed here only to get
# it under version control; not executed this session.

def load(path: Path):
    with path.open() as f:
        return json.load(f)

def classify(js, safe, high):
    if js <= safe:
        return "SAFE"
    elif js <= high:
        return "WARNING"
    else:
        return "HIGH_RISK"

def main():
    phase35 = load(Path("results/phase35_diagnostics.json"))
    phase36 = load(Path("results/phase36_diagnostics.json"))

    if not phase35:
        raise RuntimeError("Phase 3.5 diagnostics missing")

    js_vals = [d["js_divergence"] for d in phase35]
    safe = max(js_vals)
    high = safe * 2

    # Guardrail: Phase 3.5 must be SAFE
    for d in phase35:
        assert classify(d["js_divergence"], safe, high) == "SAFE"

    thresholds = {
        "thresholds": {
            "safe": safe,
            "high_risk": high,
        },
        "derived_from": "phase3.5",
        "notes": "Max JS divergence under reversible regime",
    }

    alerts = []
    for d in phase36:
        alerts.append({
            "iteration": d["iteration"],
            "risk_state": classify(
                d["js_divergence"], safe, high
            ),
            "metrics": {
                "js_divergence": d["js_divergence"],
                "tail_mass": d["tail_mass"],
            },
        })

    Path("results").mkdir(exist_ok=True)
    with open("results/phase5_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    with open("results/phase5_alerts.json", "w") as f:
        json.dump(alerts, f, indent=2)

if __name__ == "__main__":
    main()
