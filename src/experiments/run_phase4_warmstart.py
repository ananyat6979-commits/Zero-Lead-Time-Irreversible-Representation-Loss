"""
PHASE 4: STATE PERSISTENCE (WARM-START) IRREVERSIBILITY TEST

Question:
Does recovery fail when training history is preserved?

Only difference from Phase 3.5:
- Recovery is warm-started from contaminated parameters.

No architecture changes.
No data changes.
No induced bottlenecks.
"""
# NOTE (flagged, not fixed): verified directly this session that
# run_warmstart_recovery does not reset the model. It calls train on the
# already trained contaminated_model, so counts accumulate additively.
# Confirmed with a direct test: contaminated_model.total before recovery
# was 127359, recovered_model.total after recovery was 254718, exactly
# double, matching len(contaminated_tokens) + len(original_tokens) exactly.
# Also confirmed with a separate test: training a clean model twice on the
# same data, zero contamination anywhere, produces lower divergence
# (js around 0.00222) than training once (js around 0.00723), purely from
# Laplace smoothing's plus one correction mattering less at higher total
# counts. This means part of the gap between contaminated and recovered
# js_to_original values in this script's output reflects total token count
# difference, not only genuine recovery from contamination. Not resolved
# here, needs a design decision on whether the pristine floor comparison
# should also be trained twice to control for this.


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
SAMPLE_SIZE = 1000   # evaluation only (tail_mass/ttr; js_to_original/entropy now use the model's full distribution directly)
RANDOM_SEED = 42

TOKEN_PATH = Path("data/processed/pride_and_prejudice.tokens.txt")
OUT_PATH = Path("results/phase4_warmstart.json")


def load_tokens(path: Path):
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def model_distribution(model):
    """
    Returns the model's actual learned (Laplace-smoothed) distribution,
    with no sampling step -- avoids the small-sample JS-divergence
    noise floor that a stochastic sample vs. full-corpus comparison has.
    """
    return {tok: model.prob(tok) for tok in model.counts}


def evaluate(model, sampled_tokens, reference_dist):
    dist = model_distribution(model)

    return {
        "entropy": shannon_entropy(dist),
        "js_to_original": js_divergence(dist, reference_dist),
        "tail_mass": zipf_tail_mass(sampled_tokens),
        "ttr": type_token_ratio(sampled_tokens),
    }


def main():
    rng = random.Random(RANDOM_SEED)
    original_tokens = load_tokens(TOKEN_PATH)
    original_dist = empirical_distribution(original_tokens)

    results = []

    for alpha in ALPHAS:
        # ---- Contamination phase ----
        base_model = build_model("unigram")
        base_model.train(original_tokens)

        n_original = int(alpha * len(original_tokens))
        n_synthetic = len(original_tokens) - n_original
        synthetic = generate_tokens(
            model=base_model,
            seed_tokens=None,
            sample_size=n_synthetic,
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
            "contaminated": evaluate(contaminated_model, contaminated_sample, original_dist),
            "recovered": evaluate(recovered_model, recovered_sample, original_dist),
        }

        results.append(metrics)
        print(f"[alpha={alpha}] done")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"Saved results to {OUT_PATH}")


if __name__ == "__main__":
    main()