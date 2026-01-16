import random
from src.experiments.model_factory import build_model
from src.experiments.generation import generate_tokens
from src.experiments.mixture import mix_tokens


def run_self_training(original_tokens, config, model_config=None):
    rng = random.Random(config.random_seed)

    datasets = []
    current_tokens = list(original_tokens)
    datasets.append(current_tokens)

    for _ in range(config.iterations):
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

        current_tokens = mix_tokens(
            original_tokens=original_tokens,
            synthetic_tokens=synthetic,
            alpha=config.alpha,
        )

        datasets.append(current_tokens)

    return datasets