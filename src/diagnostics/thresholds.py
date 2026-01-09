import numpy as np


def compute_js_thresholds(js_values, k_safe=1.0, k_high=3.0):
    """
    Compute warning and high-risk thresholds for JS divergence
    using a reference (clean) regime.

    Parameters
    ----------
    js_values : list[float]
        JS divergences computed from reference data (e.g. D0 bootstraps)
    k_safe : float
        Number of standard deviations for SAFE → WARNING boundary
    k_high : float
        Number of standard deviations for WARNING → HIGH_RISK boundary
    """

    js_array = np.asarray(js_values, dtype=float)

    mu = js_array.mean()
    sigma = js_array.std(ddof=0)  # population std (intentional)

    return {
        "safe": mu + k_safe * sigma,
        "high_risk": mu + k_high * sigma,
    }

METRIC_THRESHOLDS = {
    "tail_mass": {
        "warning": 0.0120,
        "high_risk": 0.0123,
    },
    "js_to_D0": {
        "warning": 0.005,
        "high_risk": 0.009,
    },
}


def classify_risk(metric_name: str, value: float):
    """
    Deterministic risk classification using frozen phase 5 thresholds. 

    This is intentionally simple and conservative.
    """

    thresholds = METRIC_THRESHOLDS.get(metric_name)

    if thresholds is None:
        raise ValueError(f"No thresholds defined for metric '{metric_name}'")   
    
    if value < thresholds["warning"]:
        return "SAFE" 
    elif value < thresholds["high_risk"]:
        return "WARNING"
    else:
        return "HIGH_RISK"  