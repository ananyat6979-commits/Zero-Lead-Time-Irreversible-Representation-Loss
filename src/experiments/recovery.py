# PHASE 3.5 — IRREVERSIBILITY TEST
# Recovery under identical compute conditions.

import random
from src.experiments.model_factory import build_model


def run_recovery(clean_tokens, contaminated_tokens, config):
    """
    Retrain model on clean data after contamination.

    Returns:
    - contaminated_model
    - recovered_model
    
    Conditions:
    - Same model family
    - Same training procedure
    - No synthetic data
    - No mixing
    """

    # Model after contamination (for comparison only)
    contaminated_model = build_model(config.model_type)
    contaminated_model.train(contaminated_tokens)

    # Fresh model retrained on clean data
    recovered_model = build_model(config.model_type)
    recovered_model.train(clean_tokens)

    return contaminated_model, recovered_model
