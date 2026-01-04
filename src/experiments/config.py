# PHASE 3 LOCKED
# Causal self-training loop.
# Do not modify without invalidating downstream claims.

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    model_type: str          # "unigram" | "bigram" | "trigram"
    alpha: float             # proportion of original human data
    iterations: int          # number of self-training steps
    sample_size: int         # tokens generated per iteration
    random_seed: int         # reproducibility
