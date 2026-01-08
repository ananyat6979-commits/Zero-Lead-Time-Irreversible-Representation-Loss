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
