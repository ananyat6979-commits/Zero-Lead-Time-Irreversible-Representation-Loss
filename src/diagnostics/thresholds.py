import json
from pathlib import Path

THRESHOLD_PATH = Path("results/phase5_thresholds.json")


def load_thresholds():
    if not THRESHOLD_PATH.exists():
        raise RuntimeError(
            "Thresholds not calibrated. "
            "Run run_phase5_calibrate_thresholds.py first."
        )
    return json.loads(THRESHOLD_PATH.read_text())


def classify_risk(metric_name: str, delta_value: float):
    thresholds = load_thresholds().get(metric_name)

    if thresholds is None:
        raise ValueError(f"No thresholds defined for '{metric_name}'")

    # Verified this session: the original directional comparison
    # (delta_value < thresholds["warning"]) assumed "more negative is
    # always safer," which is backwards for tail_mass_delta. A delta of
    # exactly 0.0 read HIGH_RISK, and a delta of -0.01, a much larger
    # collapse than either calibrated threshold, read SAFE. Confirmed
    # directly against real thresholds this session.
    #
    # Fixed by classifying on distance from zero, sorting the two
    # threshold magnitudes rather than assuming warning's magnitude is
    # smaller than high_risk's. Verified this session that with real
    # calibrated thresholds, abs(warning) is actually larger than
    # abs(high_risk), so assuming an order without sorting would make
    # the WARNING band unreachable.

    magnitude = abs(delta_value)
    lo, hi = sorted([abs(thresholds["warning"]), abs(thresholds["high_risk"])])

    if magnitude <= lo:
        return "SAFE"
    elif magnitude <= hi:
        return "WARNING"
    else:
        return "HIGH_RISK"

# Verified this session: the corrected classifier genuinely discriminates.
# At alpha equals 0.1 (used by run_phase5_warmstart_alerts.py and
# run_phase6_iteration_boundary.py), real tail_mass_delta values stay
# under the calibrated tolerance, correctly reading SAFE throughout.
# At alpha equals 0.5 (results/phase36_diagnostics.json, from earlier
# this session), real tail_mass_delta values, about -0.00334 at iteration
# 1 and about -0.00353 at iteration 10, correctly read HIGH_RISK using
# this same classifier and the same calibrated thresholds. This confirms
# the calibration is not simply too tight or badly matched, it correctly
# separates a mild contamination level from a substantial one.