# PHASE 3.6 — IRREVERSIBILITY UNDER EXPLICIT BOTTLENECK

from pathlib import Path
from src.data.bottlenecks import freeze_vocabulary, apply_vocab_bottleneck
from src.experiments.self_training import run_self_training
from src.experiments.recovery import run_recovery
from src.metrics.distribution import (
    empirical_distribution,
    js_divergence,
    shannon_entropy,
    zipf_tail_mass,
    type_token_ratio,
)
from src.experiments.config import ExperimentConfig


def load_tokens(path):
    return path.read_text(encoding="utf-8", errors="replace").splitlines()

def main():
    # ---- CONFIG (LOCKED) ----
    config = ExperimentConfig(
        model_type="unigram",
        alpha=0.5,
        iterations=10,
        sample_size=20,
        random_seed=42,
    )

    # ---- LOAD ORIGINAL DATA ----
    token_path = Path("data/processed/pride_and_prejudice.tokens.txt")
    original_tokens = load_tokens(token_path)

    # ---- APPLY BOTTLENECK ----
    vocab = freeze_vocabulary(original_tokens, max_vocab_size=None)
    original_b = apply_vocab_bottleneck(original_tokens, vocab)

    # ---- PHASE 3 UNDER BOTTLENECK ----
    datasets = run_self_training(original_b, config)
    contaminated = datasets[-1]

    # ---- PHASE 3.5 RECOVERY UNDER BOTTLENECK ----
    contaminated_model, recovered_model = run_recovery(
        original_b, contaminated, config
    )

    # ---- DISTRIBUTIONAL COMPARISON ----
    original_dist = empirical_distribution(original_b)
    contaminated_dist = empirical_distribution(contaminated)

    recovered_tokens = apply_vocab_bottleneck(original_tokens, vocab)
    recovered_dist = empirical_distribution(recovered_tokens)

    results = {
        "entropy_original": shannon_entropy(original_dist),
        "entropy_contaminated": shannon_entropy(contaminated_dist),
        "entropy_recovered": shannon_entropy(recovered_dist),
        "js_contaminated": js_divergence(original_dist, contaminated_dist),
        "js_recovered": js_divergence(original_dist, recovered_dist),
        "tail_contaminated": zipf_tail_mass(contaminated),
        "tail_recovered": zipf_tail_mass(recovered_tokens),
        "ttr_contaminated": type_token_ratio(contaminated),
        "ttr_recovered": type_token_ratio(recovered_tokens),
    }

    print(results)


if __name__ == "__main__":
    main()
