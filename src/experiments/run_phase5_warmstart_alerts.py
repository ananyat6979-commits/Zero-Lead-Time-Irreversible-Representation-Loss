"""
PHASE 5′ — EARLY-WARNING DIAGNOSTICS UNDER STATE PERSISTENCE

This phase tests whether distributional diagnostics provide
advance warning when degradation accumulates through training history.

Key properties:
- Warm-start training (state is preserved)
- No architectural changes
- Frozen thresholds (calibrated from D0)
- No smoothing, no collapse assumptions
"""

import json
from pathlib import Path

from src.experiments.model_factory import build_model
from src.experiments.generation import generate_tokens
from src.experiments.mixture import mix_tokens
from src.experiments.config import ExperimentConfig
import random


from src.metrics.distribution import (
    empirical_distribution,
    js_divergence,
    zipf_tail_mass,
    shannon_entropy,
    type_token_ratio,
)

from src.diagnostics.thresholds import classify_risk


# ---------------------------------------------------------------------
# CONFIG — LOCKED
# ---------------------------------------------------------------------

MODEL_TYPE = "unigram"
ALPHA = 0.1                 # fixed contamination
SAMPLE_SIZE = 20
MAX_ITER = 15
RANDOM_SEED = 42

TOKEN_PATH = Path("data/processed/pride_and_prejudice.tokens.txt")
OUT_PATH = Path("results/phase5_warmstart_alerts.json")


# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------

def load_tokens(path: Path):
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def compute_metrics(tokens, reference_dist):
    dist = empirical_distribution(tokens)
    return {
        "js_to_D0": js_divergence(dist, reference_dist),
        "entropy": shannon_entropy(dist),
        "tail_mass": zipf_tail_mass(tokens),
        "ttr": type_token_ratio(tokens),
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    rng = random.Random(RANDOM_SEED)
    original_tokens = load_tokens(TOKEN_PATH)
    reference_dist = empirical_distribution(original_tokens)

    # ---- Initialize model ONCE (state persistence) ----
    model = build_model(MODEL_TYPE)
    model.train(original_tokens)

    base_tail_mass = zipf_tail_mass(original_tokens)

    results = []

    for t in range(MAX_ITER + 1):
        # ---- Generate from current model state ----
        synthetic = generate_tokens(
            model=model,
            seed_tokens=None,
            sample_size=SAMPLE_SIZE,
            rng=rng,
        )

        # ---- Mix data (fixed α) ----
        mixed_tokens = list(
            mix_tokens(
                original_tokens=original_tokens,
                synthetic_tokens=synthetic,
                alpha=ALPHA,
            )
        )

        # ---- Diagnostics BEFORE updating state ----
        metrics = compute_metrics(synthetic, reference_dist)

        tail_delta = metrics["tail_mass"] - base_tail_mass
        risk = classify_risk("tail_mass_delta", tail_delta)

        record = {
            "iteration": t,
            "alpha": ALPHA,
            "num_synthetic_tokens": len(synthetic),
            **metrics,
            "risk_state": risk,
        }

        results.append(record)

        print(
            f"[D{t}] "
            f"tail={metrics['tail_mass']:.6f} | "
            f"JS={metrics['js_to_D0']:.6f} → {risk}"
        )

        # ---- Warm-start update (THIS is the causal change) ----
        model.train(mixed_tokens)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved Phase 5′ warm-start alerts to {OUT_PATH}")


if __name__ == "__main__":
    main()
