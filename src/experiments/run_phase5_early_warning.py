import json
from pathlib import Path

# NOTE (collision fixed, method choice still open): this script previously
# wrote to results/phase5_thresholds.json, the same path used by
# run_phase5_calibrate_thresholds.py, with an incompatible schema. This
# derives {"thresholds": {"safe": ..., "high_risk": ...}} from
# max(phase3.5's raw js_divergence) * 2, while
# run_phase5_calibrate_thresholds.py derives {"tail_mass_delta": {"warning":
# ..., "high_risk": ...}} via bootstrap resampling. These are still NOT
# interchangeable, run_phase5_warmstart_alerts.py and
# run_phase6_iteration_boundary.py both require the tail_mass_delta schema
# from run_phase5_calibrate_thresholds.py, not this script's output.
# The output path is now renamed to results/phase5_thresholds_js_based.json
# so the two scripts can no longer silently overwrite each other's file.
# Which calibration method should actually be canonical is still an open
# design decision, not resolved here. Still not run this session, since
# results/phase5_alerts.json's schema also needs reconciling with
# plot_phase5_alerts.py before this is fully safe to exercise end to end.

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
    with open("results/phase5_thresholds_js_based.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    with open("results/phase5_alerts_js_based.json", "w") as f:
        json.dump(alerts, f, indent=2)

if __name__ == "__main__":
    main()
