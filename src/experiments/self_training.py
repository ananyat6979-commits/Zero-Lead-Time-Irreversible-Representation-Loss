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
    model_config=None,
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
        model = build_model(
            config.model_type,
            model_config=model_config, 
)
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
# Overwrite semantics are intentional:
# each iteration replaces the dataset rather than accumulating tokens.
# This prevents uncontrolled growth and isolates data-feedback effects.
        current_tokens = mixed
        datasets.append(current_tokens)

    return datasets
