"""
PHASE 5.26 — DELTA-BASED THRESHOLD CALIBRATION (LOCKED)

Calibrates WARNING / HIGH_RISK thresholds on
Δ(metric) relative to the clean reference distribution (D0).
"""

import json
import random
from pathlib import Path
from src.metrics.distribution import zipf_tail_mass

DATA_PATH = Path("data/processed/pride_and_prejudice.tokens.txt")
OUT_PATH = Path("results/phase5_thresholds.json")

N_BOOT = 50
K_WARNING = 1.0
K_HIGH = 3.0
RNG = random.Random(42)


def load_tokens(p):
    return p.read_text(encoding="utf-8", errors="replace").splitlines()


def mean_std(xs):
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / len(xs)
    return mu, var ** 0.5


def main():
    tokens = load_tokens(DATA_PATH)
    n = len(tokens)

    base_tail = zipf_tail_mass(tokens)

    deltas = []
    for _ in range(N_BOOT):
        sample = [tokens[RNG.randrange(n)] for _ in range(n)]
        deltas.append(zipf_tail_mass(sample) - base_tail)

    mu, sd = mean_std(deltas)

    thresholds = {
        "tail_mass_delta": {
            "warning": mu + K_WARNING * sd,
            "high_risk": mu + K_HIGH * sd,
        }
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(thresholds, indent=2))
    print(f"Saved calibrated delta thresholds to {OUT_PATH}")


if __name__ == "__main__":
    main()
