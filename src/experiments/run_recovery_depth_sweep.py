import random
import copy
from pathlib import Path

from src.experiments.model_factory import build_model
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
    original_dist = empirical_distribution(original)

    contaminated_model = build_model("unigram")
    contaminated_model.train(contaminated)

    # NOTE: previously this sweep was a no-op, config.iterations was
    # never consumed inside run_recovery(). Redesigned to genuinely vary
    # with recovery_iters, real warm-start recovery, calling train()
    # repeatedly on a deep copy of the contaminated model. A matched-count
    # pristine control is trained the same number of times on clean data
    # only, so the training-call-count artifact verified earlier this
    # session, training twice lowers divergence purely from Laplace
    # smoothing regardless of content, is present in both arms and
    # js_recovered_minus_control isolates genuine recovery from that
    # artifact.
    results = []
    for recovery_iters in [1, 2, 5, 10, 20, 50]:
        recovered = copy.deepcopy(contaminated_model)
        pristine_control = build_model("unigram")

        for _ in range(recovery_iters):
            recovered.train(original)
            pristine_control.train(original)

        recovered_tokens = recovered.sample(
            sample_size=len(original),
            rng=random.Random(42),
        )
        pristine_tokens = pristine_control.sample(
            sample_size=len(original),
            rng=random.Random(42),
        )

        js_recovered = js_divergence(
            original_dist,
            empirical_distribution(recovered_tokens),
        )
        js_pristine_control = js_divergence(
            original_dist,
            empirical_distribution(pristine_tokens),
        )

        results.append({
            "recovery_iters": recovery_iters,
            "js_recovered": js_recovered,
            "js_pristine_control": js_pristine_control,
            "js_recovered_minus_control": js_recovered - js_pristine_control,
            "tail_mass_recovered": zipf_tail_mass(recovered_tokens),
        })

    write_json(results, Path("results/recovery_depth_sweep.json"))


if __name__ == "__main__":
    main()