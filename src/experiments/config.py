from dataclasses import dataclass

@dataclass(frozen=True)
class ExperimentConfig:
    model_type: str
    alpha: float
    iterations: int
    sample_size: int
    random_seed: int
