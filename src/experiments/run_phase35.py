from pathlib import Path
from src.metrics.distribution import empirical_distribution, js_divergence, zipf_tail_mass
from src.experiments.config import ExperimentConfig
from src.experiments.utils import write_diagnostics

def load_tokens(path: Path):
    return path.read_text(encoding="utf-8").splitlines()

def main():
    base = Path("data/generated/phase3")
    d0 = load_tokens(base / "D0.txt")
    d0_dist = empirical_distribution(d0)

    diagnostics = []

    for k in [3, 5, 7, 10]:
        contaminated = load_tokens(base / f"D{k}.txt")
        contaminated_dist = empirical_distribution(contaminated)

        diagnostics.append({
            "iteration": f"k={k}",
            "js_divergence": js_divergence(d0_dist, contaminated_dist),
            "tail_mass": zipf_tail_mass(contaminated),
        })

    write_diagnostics(
        diagnostics,
        Path("results/phase35_diagnostics.json")
    )

if __name__ == "__main__":
    main()
