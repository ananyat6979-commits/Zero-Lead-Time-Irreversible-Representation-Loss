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
    
    # NOTE (held, not fixed): config.iterations is
    # never consumed anywhere inside run_recovery(), so this sweep is
    # currently a no-op. Verified empirically: every recovery_iters value
    # below produces bit-identical output (js_recovered=0.027955425229532538
    # regardless of recovery_iters). Needs a design decision on what
    # "recovery iterations" should actually mean before this is fixed,
    # not fixed here.
    results = []

    for recovery_iters in [1, 2, 5, 10, 20, 50]:
        config = ExperimentConfig(
            model_type="unigram",
            alpha=0.5,
            iterations=recovery_iters,
            sample_size=20,
            random_seed=42,
        )

        _, recovered_model = run_recovery(
            original,
            contaminated,
            config,
            model_config=None,  # 🔓 NO constraint
        )

        recovered_tokens = recovered_model.sample(
            sample_size=len(original),
            rng=random.Random(config.random_seed),
        )

        results.append({
            "recovery_iters": recovery_iters,
            "js_recovered": js_divergence(
                empirical_distribution(original),
                empirical_distribution(recovered_tokens),
            ),
            "tail_mass_recovered": zipf_tail_mass(recovered_tokens),
        })

    write_json(results, Path("results/recovery_depth_sweep.json"))


if __name__ == "__main__":
    main()
