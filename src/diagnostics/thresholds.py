import numpy as np
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


def classify_risk(metric_name: str, value: float):
    thresholds = load_thresholds().get(metric_name)

    if thresholds is None:
        raise ValueError(f"No thresholds for metric '{metric_name}'")

    if value < thresholds["warning"]:
        return "SAFE"
    elif value < thresholds["high_risk"]:
        return "WARNING"
    else:
        return "HIGH_RISK"