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
