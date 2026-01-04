from src.models.ngram import UnigramLM, BigramLM, TrigramLM


def build_model(model_type: str):
    if model_type == "unigram":
        return UnigramLM()
    if model_type == "bigram":
        return BigramLM()
    if model_type == "trigram":
        return TrigramLM()

    raise ValueError(f"Unknown model type: {model_type}")
