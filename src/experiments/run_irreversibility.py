# PHASE 3.5
# Irreversibility (hysteresis) experiment runner.
# Demonstrates path dependence under self-training.
#
# No metrics. No plots. Dataset generation only.

from src.experiments.self_training import run_self_training
from src.experiments.recovery import run_recovery
from src.experiments.config import ExperimentConfig


def run_irreversibility_experiment(
    original_tokens,
    config: ExperimentConfig,
):
    """
    Runs forward self-training followed by recovery training.

    Returns:
        {
            "clean": D0,
            "contaminated": D_k,
            "recovered": D_recovered
        }
    """

    # --- Forward contamination ---
    datasets = run_self_training(
        original_tokens=original_tokens,
        config=config,
    )

    D0 = datasets[0]
    Dk = datasets[-1]

    # --- Recovery phase ---
    D_recovered = run_recovery(
        clean_tokens=D0,
        contaminated_tokens=Dk,
        config=config,
    )

    return {
        "clean": D0,
        "contaminated": Dk,
        "recovered": D_recovered,
    }
