from src.models.ngram import UnigramLM

def build_model(model_type, model_config=None):
    model_config = model_config or {}

    if model_type == "unigram":
        return UnigramLM(
            min_token_count=model_config.get("min_token_count")
        )

    raise ValueError(f"Unknown model type: {model_type}")
