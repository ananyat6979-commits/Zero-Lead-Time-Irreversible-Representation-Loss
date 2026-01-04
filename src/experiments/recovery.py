from src.experiments.model_factory import build_model
from src.experiments.generation import generate_tokens
from src.experiments.mixture import mix_tokens


def run_recovery(clean_tokens, contaminated_tokens, config):
    """
    Recovery training after contamination.

    Uses the same compute budget and data size as forward training.
    """

    model = build_model(config.model_type)
    model.train(contaminated_tokens)

    synthetic = generate_tokens(
        model=model,
        seed_tokens=None,
        sample_size=config.sample_size,
        rng=None,  # RNG is irrelevant here; recovery is evaluated statistically later
    )

    recovered = mix_tokens(
        original_tokens=clean_tokens,
        synthetic_tokens=synthetic,
        alpha=config.alpha,
    )

    return list(recovered)
