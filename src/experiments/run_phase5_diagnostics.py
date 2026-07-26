"""
PHASE 5 — EARLY WARNING DIAGNOSTICS

This phase validates whether distributional metrics provide
early warning of self-training degradation *before* collapse
or irreversibility is observable.

NO training.
NO recovery.
NO new mechanisms.

Pure diagnosis.
"""

import json
from pathlib import Path

from src.metrics.distribution import (
    empirical_distribution,
    shannon_entropy,
    js_divergence,
    zipf_tail_mass,
    type_token_ratio,
)

# ------------------
# CONFIG (LOCKED)
# ------------------

PHASE3_DIR = Path("data/generated/phase3")
OUT_PATH = Path("results/phase5_diagnostics.json")

ITERATIONS = list(range(0, 11))  # D0 ... D10
TAIL_FRACTION = 0.1


def load_tokens(path: Path):
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def compute_metrics(tokens, reference_dist):
    dist = empirical_distribution(tokens)

    return {
        "entropy": shannon_entropy(dist),
        "js_to_D0": js_divergence(dist, reference_dist),
        "tail_mass": zipf_tail_mass(tokens, tail_fraction=TAIL_FRACTION),
        "ttr": type_token_ratio(tokens),
    }


def main():
    # ---- Load baseline (D0) ----
    d0_path = PHASE3_DIR / "D0.txt"
    if not d0_path.exists():
        raise FileNotFoundError("D0.txt not found. Phase 3 must be complete.")

    d0_tokens = load_tokens(d0_path)
    d0_dist = empirical_distribution(d0_tokens)

    results = []

    # ---- Iterate through self-training steps ----
    for k in ITERATIONS:
        path = PHASE3_DIR / f"D{k}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset: {path}")

        tokens = load_tokens(path)
        metrics = compute_metrics(tokens, d0_dist)

        record = {
            "iteration": k,
            "num_tokens": len(tokens),
            **metrics,
        }

        results.append(record)
        print(f"[D{k}] metrics computed")

    # ---- Save ----
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"Saved Phase 5 diagnostics to {OUT_PATH}")


if __name__ == "__main__":
    main()
