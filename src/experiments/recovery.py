from src.experiments.model_factory import build_model

def run_recovery(original_tokens, contaminated_tokens, config, model_config=None):
    contaminated_model = build_model(
        config.model_type,
        model_config=model_config,
    )
    contaminated_model.train(contaminated_tokens)

    recovered_model = build_model(
        config.model_type,
        model_config=model_config,
    )
    recovered_model.train(original_tokens)

    return contaminated_model, recovered_model
