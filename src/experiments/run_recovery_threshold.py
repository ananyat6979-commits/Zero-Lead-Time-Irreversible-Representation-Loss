# src/experiments/run_recovery_threshold.py

import random
from pathlib import Path

from src.experiments.self_training import run_self_training
from src.experiments.recovery import run_recovery
from src.metrics.distribution import (
    empirical_distribution,
    js_divergence,
    zipf_tail_mass,
)
from src.experiments.config import ExperimentConfig
from src.experiments.utils import write_json


def load_tokens(path: Path):
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

    # ---- LOAD CLEAN DATA ----
    token_path = Path("data/processed/pride_and_prejudice.tokens.txt")
    original_tokens = load_tokens(token_path)
    original_dist = empirical_distribution(original_tokens)

    results = []

    # Sweep contamination depth
    for k in [1, 2, 3, 5, 7, 10]:
        # ---- PHASE 3: SELF-TRAINING UNDER CONSTRAINT ----
        datasets = run_self_training(
            original_tokens,
            config,
            model_config={"min_token_count": 3},  # irreversibility trigger
        )

        contaminated = datasets[k]

        # ---- RECOVERY FROM CLEAN DATA ----
        _, recovered_model = run_recovery(
            original_tokens,
            contaminated,
            config,
            model_config={"min_token_count": 3},
	    retrain_from_contaminated=True,
        )

        recovered_tokens = recovered_model.sample(
            sample_size=len(original_tokens),
            rng=random.Random(config.random_seed),
        )

        recovered_dist = empirical_distribution(recovered_tokens)

        results.append({
            "k": k,
            "js_recovered": js_divergence(original_dist, recovered_dist),
            "tail_mass_recovered": zipf_tail_mass(recovered_tokens),
        })

    write_json(results, Path("results/recovery_threshold.json"))


if __name__ == "__main__":
    main()
