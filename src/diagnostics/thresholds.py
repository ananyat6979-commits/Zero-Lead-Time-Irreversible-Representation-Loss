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

    if delta_value < thresholds["warning"]:
        return "SAFE"
    elif delta_value < thresholds["high_risk"]:
        return "WARNING"
    else:
        return "HIGH_RISK"
