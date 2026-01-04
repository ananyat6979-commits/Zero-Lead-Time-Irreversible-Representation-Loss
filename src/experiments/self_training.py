# PHASE 3 LOCKED
# Causal self-training loop.
# Do not modify without invalidating downstream claims.


import random
from src.experiments.model_factory import build_model
from src.experiments.generation import generate_tokens
from src.experiments.mixture import mix_tokens


def run_self_training(
    original_tokens,
    config,
):
    """
    Runs a controlled self-training loop.

    Returns a list of token sequences:
    [D0, D1, D2, ...]
    """
    rng = random.Random(config.random_seed)

    datasets = []
    current_tokens = list(original_tokens)

    datasets.append(current_tokens)

    for step in range(config.iterations):
        model = build_model(config.model_type)
        model.train(current_tokens)

        synthetic = generate_tokens(
            model=model,
            seed_tokens=None,
            sample_size=config.sample_size,
            rng=rng,
        )

        mixed = mix_tokens(
            original_tokens=original_tokens,
            synthetic_tokens=synthetic,
            alpha=config.alpha,
        )

        current_tokens = mixed
        datasets.append(current_tokens)

    return datasets
