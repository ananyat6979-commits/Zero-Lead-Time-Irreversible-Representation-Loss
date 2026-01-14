def generate_tokens(model, seed_tokens, sample_size, rng):
    if rng is None:
        raise ValueError("RNG must be provided for stochastic generation")
    return model.sample(
        sample_size=sample_size,
        rng=rng
    )
