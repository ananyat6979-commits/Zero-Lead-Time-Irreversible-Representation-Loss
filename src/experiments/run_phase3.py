# PHASE 3
# Deterministic self-training runner (Unigram only).
# Produces D0 -> Dk datasets under controlled contamination.

from pathlib import Path

from src.experiments.config import ExperimentConfig
from src.experiments.self_training import run_self_training


def load_tokens(path: Path):
    # Tokens are already materialized; decoding must be permissive.
    # Encoding issues are NOT part of Phase 3 claims.
    return path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()


def save_dataset(tokens, path):
    # Explicit UTF-8 to avoid Windows default encoding issues
    path.write_text(
        "\n".join(tokens),
        encoding="utf-8",
        errors="replace",
    )


def main():
    # ---- INPUT DATA ----
    token_path = Path("data/processed/pride_and_prejudice.tokens.txt")
    original_tokens = load_tokens(token_path)

    # ---- EXPERIMENT CONFIG (LOCKED) ----
    config = ExperimentConfig(
        model_type="unigram",
        alpha=0.8,
        iterations=10,
        sample_size=20,
        random_seed=42,
    )

    # ---- RUN SELF-TRAINING ----
    datasets = run_self_training(
        original_tokens=original_tokens,
        config=config,
    )

    # ---- SAVE OUTPUTS ----
    out_dir = Path("data/generated/phase3")

    for i, tokens in enumerate(datasets):
        save_dataset(tokens, out_dir / f"D{i}.txt")

    print(f"Phase 3 complete. Generated {len(datasets)} datasets.")


if __name__ == "__main__":
    main()
