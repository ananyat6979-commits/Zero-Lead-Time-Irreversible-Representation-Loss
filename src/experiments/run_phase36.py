import random
from pathlib import Path
from src.experiments.self_training import run_self_training
from src.metrics.distribution import empirical_distribution, js_divergence, zipf_tail_mass
from src.experiments.config import ExperimentConfig
from src.experiments.utils import write_diagnostics

def load_tokens(path):
    return path.read_text(
        encoding="utf-8",
        errors="replace"
    ).splitlines()


def main():
    config = ExperimentConfig(
        model_type="unigram",
        alpha=0.5,           # SAME AS PHASE 3.5
        iterations=10,
        sample_size=20,
        random_seed=42,
    )

    original = load_tokens(
        Path("data/processed/pride_and_prejudice.tokens.txt")
    )
    original_dist = empirical_distribution(original)

    model_config = {"min_token_count": 3}

    datasets = run_self_training(
        original,
        config,
        model_config=model_config,
    )

    diagnostics = []
    for i, tokens in enumerate(datasets):
        diagnostics.append({
            "iteration": i,
            "js_divergence": js_divergence(
                original_dist,
                empirical_distribution(tokens),
            ),
            "tail_mass": zipf_tail_mass(tokens),
        })

    write_diagnostics(
        diagnostics,
        Path("results/phase36_diagnostics.json")
    )

if __name__ == "__main__":
    main()
