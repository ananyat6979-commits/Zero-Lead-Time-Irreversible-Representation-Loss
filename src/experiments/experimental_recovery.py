# PHASE 3.5 — IRREVERSIBILITY TEST
# Recovery under identical compute conditions.

import copy
import random
from src.experiments.model_factory import build_model

def run_recovery_phase35(clean_tokens, contaminated_tokens, config):
    """
    Phase 3.5 recovery: unconstrained, memoryless retraining.
    This function is intentionally separate to preserve negative control.
    """
    contaminated_model = build_model(config.model_type)
    contaminated_model.train(contaminated_tokens)

    recovered_model = build_model(config.model_type)
    recovered_model.train(clean_tokens)

    return contaminated_model, recovered_model


def run_reset_recovery(clean_tokens, config):
    """
    Baseline recovery: retrain from scratch on clean data.
    Proven reversible regime.
    """
    model = build_model(config.model_type)
    model.train(clean_tokens)
    return model


def run_warmstart_recovery(contaminated_model, clean_tokens):
    """
     Phase 4 recovery: warm-start retraining from contaminated state.
    This introduces state persistence without changing:
    - data
    - model class
    - training procedure

    Returns a NEW model object (deep copy of contaminated_model, then
    trained further on clean_tokens) so that the caller's original
    contaminated_model is left untouched and can still be evaluated
    in its pre-recovery state.

    """
    recovered_model = copy.deepcopy(contaminated_model)
    recovered_model.train(clean_tokens)
    return recovered_model


def run_recovery(original_tokens, contaminated_tokens, config, model_config=None):
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
    contaminated_model = build_model(
        config.model_type,
        model_config=model_config,
    )
    contaminated_model.train(contaminated_tokens)

    # Fresh model retrained on clean data
    recovered_model = build_model(
        config.model_type,
        model_config=model_config,
    )
    recovered_model.train(original_tokens)

    return contaminated_model, recovered_model

    
