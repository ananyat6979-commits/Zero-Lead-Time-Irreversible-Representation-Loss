from pathlib import Path
from src.experiments.recovery import run_recovery
from src.metrics.distribution import (
    empirical_distribution,
    shannon_entropy,
    js_divergence,
    zipf_tail_mass,
    type_token_ratio,
)
from src.experiments.config import ExperimentConfig


def load_tokens(path: Path):
    return path.read_text(encoding="utf-8").splitlines()


def main():
    base_dir = Path("data/generated/phase3")
    d0_tokens = load_tokens(base_dir / "D0.txt")

    results = []

    for k in [3, 5, 7, 10]:
        contaminated = load_tokens(base_dir / f"D{k}.txt")

        config = ExperimentConfig(
            model_type="unigram",
            alpha=0.8,
            iterations=10,
            sample_size=20,
            random_seed=42,
        )

        recovered_model = run_recovery(
            clean_tokens=d0_tokens,
            contaminated_tokens=contaminated,
            config=config,
        )

        recovered_dist = empirical_distribution(d0_tokens)
        original_dist = empirical_distribution(d0_tokens)
        contaminated_dist = empirical_distribution(contaminated)

        results.append({
            "k": k,
            "entropy_contaminated": shannon_entropy(contaminated_dist),
            "entropy_recovered": shannon_entropy(recovered_dist),
            "js_contaminated": js_divergence(original_dist, contaminated_dist),
            "js_recovered": js_divergence(original_dist, recovered_dist),
            "tail_contaminated": zipf_tail_mass(contaminated),
            "tail_recovered": zipf_tail_mass(d0_tokens),
            "ttr_contaminated": type_token_ratio(contaminated),
            "ttr_recovered": type_token_ratio(d0_tokens),
        })

    for row in results:
        print(row)


if __name__ == "__main__":
    main()
