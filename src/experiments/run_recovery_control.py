# src/experiments/run_recovery_control.py

import random
from pathlib import Path

from src.experiments.recovery import run_recovery
from src.experiments.config import ExperimentConfig
from src.metrics.distribution import (
    empirical_distribution,
    js_divergence,
    zipf_tail_mass,
)
from src.experiments.utils import write_json


def load_tokens(path: Path):
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def main():
    base_dir = Path("data/generated/phase3")
    original = load_tokens(base_dir / "D0.txt")
    contaminated = load_tokens(base_dir / "D10.txt")

    config = ExperimentConfig(
        model_type="unigram",
        alpha=0.5,
        iterations=10,
        sample_size=20,
        random_seed=42,
    )

    # CONTROL: NO representation constraint
    # NOTE: with retrain_from_contaminated left at its default (False),
    # recovered_model trains purely on `original` and never touches
    # `contaminated` at all. Verified empirically: swapping contaminated
    # for a copy of `original` produces bit-identical output
    # (js_recovered=0.027955425229532538 either way). This script
    # therefore measures the pristine, zero-contamination sampling
    # floor, not recovery from any specific contamination depth.
    # See run_recovery_threshold.py for the k-dependent version.

    contaminated_model, recovered_model = run_recovery(
        original,
        contaminated,
        config,
        model_config=None,
    )

    recovered_tokens = recovered_model.sample(
        sample_size=len(original),
        rng=random.Random(config.random_seed),
    )

    result = {
        "js_recovered": js_divergence(
            empirical_distribution(original),
            empirical_distribution(recovered_tokens),
        ),
        "tail_mass_recovered": zipf_tail_mass(recovered_tokens),
    }

    write_json(result, Path("results/recovery_control.json"))


if __name__ == "__main__":
    main()
