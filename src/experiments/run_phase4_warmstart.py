"""
PHASE 4 — STATE PERSISTENCE (WARM-START) IRREVERSIBILITY TEST

Question:
Does recovery fail when training history is preserved?

Only difference from Phase 3.5:
- Recovery is warm-started from contaminated parameters.

No architecture changes.
No data changes.
No induced bottlenecks.
"""

import json
import random
from pathlib import Path

from src.experiments.model_factory import build_model
from src.experiments.generation import generate_tokens
from src.experiments.mixture import mix_tokens
from src.experiments.experimental_recovery import run_warmstart_recovery
from src.experiments.config import ExperimentConfig
import random


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

ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]
SAMPLE_SIZE = 1000   # evaluation only
RANDOM_SEED = 42

TOKEN_PATH = Path("data/processed/pride_and_prejudice.tokens.txt")
OUT_PATH = Path("results/phase4_warmstart.json")


def load_tokens(path: Path):
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def evaluate(sampled_tokens, reference_tokens):
    p = empirical_distribution(sampled_tokens)
    q = empirical_distribution(reference_tokens)

    return {
        "entropy": shannon_entropy(p),
        "js_to_original": js_divergence(p, q),
        "tail_mass": zipf_tail_mass(sampled_tokens),
        "ttr": type_token_ratio(sampled_tokens),
    }


def main():
    rng = random.Random(RANDOM_SEED)
    original_tokens = load_tokens(TOKEN_PATH)

    results = []

    for alpha in ALPHAS:
        # ---- Contamination phase ----
        base_model = build_model("unigram")
        base_model.train(original_tokens)

        synthetic = generate_tokens(
            model=base_model,
            seed_tokens=None,
            sample_size=SAMPLE_SIZE,
            rng=rng,
        )

        contaminated_tokens = mix_tokens(
            original_tokens=original_tokens,
            synthetic_tokens=synthetic,
            alpha=alpha,
        )

        contaminated_model = build_model("unigram")
        contaminated_model.train(contaminated_tokens)

        # ---- Warm-start recovery ----
        recovered_model = run_warmstart_recovery(
            contaminated_model,
            original_tokens,
        )

        contaminated_sample = generate_tokens(
            model=contaminated_model,
            seed_tokens=None,
            sample_size=SAMPLE_SIZE,
            rng=rng,
        )

        recovered_sample = generate_tokens(
            model=recovered_model,
            seed_tokens=None,
            sample_size=SAMPLE_SIZE,
            rng=rng,
        )

        metrics = {
            "alpha": alpha,
            "contaminated": evaluate(contaminated_sample, original_tokens),
            "recovered": evaluate(recovered_sample, original_tokens),
        }

        results.append(metrics)
        print(f"[alpha={alpha}] done")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"Saved results to {OUT_PATH}")


if __name__ == "__main__":
    main()
