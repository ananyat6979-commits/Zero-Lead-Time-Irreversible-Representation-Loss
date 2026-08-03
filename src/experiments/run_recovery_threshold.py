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

        # ---- Matched-count pristine control ----
        # recovered_model above trains on contaminated_tokens + original_tokens
        # (retrain_from_contaminated=True), i.e. two effective passes worth of
        # tokens. Per the same artifact already found and controlled for in
        # run_phase4_warmstart.py and run_recovery_depth_sweep.py: training on
        # more total tokens lowers JS-divergence via Laplace smoothing alone,
        # independent of any real recovery from contamination. This control
        # trains a fresh model on original_tokens twice, matching recovered_model's
        # total token count, so the artifact is present in both and cancels
        # out in the comparison below.
        from src.experiments.model_factory import build_model

        pristine_control = build_model(config.model_type, model_config={"min_token_count": 3})
        pristine_control.train(original_tokens)
        pristine_control.train(original_tokens)

        control_tokens = pristine_control.sample(
            sample_size=len(original_tokens),
            rng=random.Random(config.random_seed),
        )
        control_dist = empirical_distribution(control_tokens)
        js_control = js_divergence(original_dist, control_dist)

        results.append({
            "k": k,
            "js_recovered": js_divergence(original_dist, recovered_dist),
            "tail_mass_recovered": zipf_tail_mass(recovered_tokens),
            "js_control_matched_count": js_control,
            "js_recovered_minus_control": (
                js_divergence(original_dist, recovered_dist) - js_control
            ),
        })

    write_json(results, Path("results/recovery_threshold.json"))


if __name__ == "__main__":
    main()