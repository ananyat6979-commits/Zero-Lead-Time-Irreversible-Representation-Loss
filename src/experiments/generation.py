# PHASE 3 LOCKED
# Causal self-training loop.
# Do not modify without invalidating downstream claims.

import random


def generate_tokens(model, seed_tokens, sample_size, rng: random.Random):
    """
    Generate synthetic tokens using the provided model.
    """
    if hasattr(model, "sample"):
        return model.sample(
            sample_size=sample_size,
            rng=rng
        )

    raise RuntimeError("Model does not support sampling")

