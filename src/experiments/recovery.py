from src.experiments.model_factory import build_model


def run_recovery(original_tokens, contaminated_tokens, config, model_config=None, retrain_from_contaminated=False):
    contaminated_model = build_model(
        config.model_type,
        model_config=model_config,
    )
    contaminated_model.train(contaminated_tokens)

    recovered_model = build_model(
        config.model_type,
        model_config=model_config,
    )
    if retrain_from_contaminated:
        recovered_model.train(contaminated_tokens + original_tokens)
    else:
        recovered_model.train(original_tokens)

    return contaminated_model, recovered_model
