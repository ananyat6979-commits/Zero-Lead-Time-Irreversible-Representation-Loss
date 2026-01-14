from src.models.ngram import UnigramLM, BigramLM, TrigramLM


def build_model(model_type, model_config=None):
    model_config = model_config or {}

    if model_type == "unigram":
        return UnigramLM(
            min_token_count=model_config.get("min_token_count")
        )
    if model_type == "bigram":
        return BigramLM()
    if model_type == "trigram":
        return TrigramLM()

    raise ValueError(f"Unknown model type: {model_type}")
