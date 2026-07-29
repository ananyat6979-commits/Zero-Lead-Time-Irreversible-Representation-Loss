"""
PHASE 6A: ITERATION BOUNDARY

Question:
How long can a self-training pipeline run before entering a risk regime,
under fixed contamination and validated early-warning diagnostics?

Independent variable:
- Number of self-training iterations

Everything else is held constant.
"""

import json
from pathlib import Path
import random

from src.experiments.model_factory import build_model
from src.experiments.generation import generate_tokens
from src.experiments.mixture import mix_tokens
from src.diagnostics.thresholds import classify_risk
from src.metrics.distribution import zipf_tail_mass
from src.experiments.config import ExperimentConfig
import random


# ------------------
# CONFIG (LOCKED)
# ------------------

MODEL_TYPE = "unigram"
ALPHA = 0.1                  # same α that triggered risk in Phase 5
SAMPLE_SIZE = 20
RANDOM_SEED = 42

MAX_ITER = 30

SAT_EPS = 1e-5               # saturation tolerance
SAT_WINDOW = 5               # consecutive flat iterations

TOKEN_PATH = Path("data/processed/pride_and_prejudice.tokens.txt")
OUT_PATH = Path("results/phase6_iteration_boundary.json")


# ------------------
# HELPERS
# ------------------

def load_tokens(path: Path):
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


# ------------------
# MAIN
# ------------------

def main():
    rng = random.Random(RANDOM_SEED)

    original_tokens = load_tokens(TOKEN_PATH)
    current_tokens = list(original_tokens)
    base_tail_mass = zipf_tail_mass(original_tokens)

    # Fixed: previously a fresh, untrained model was built every single
    # iteration inside the loop, model = build_model(MODEL_TYPE), trained
    # only on current_tokens, so there was no real warm-start persistence
    # across iterations at all, despite the script's own docstring
    # stating everything except iteration count is held constant.
    # Verified this session: after correcting the calibration thresholds,
    # this memoryless version produced noisy, repeatedly flipping
    # SAFE/WARNING/HIGH_RISK classifications across iterations, unlike
    # run_phase5_warmstart_alerts.py's genuine warm-start pattern, which
    # showed a clean one-time transition. Fixed by building the model
    # once, outside the loop, and continuing to train the same object
    # each iteration, matching run_phase5_warmstart_alerts.py's actual
    # warm-start mechanism.
    model = build_model(MODEL_TYPE)
    model.train(original_tokens)

    history = []
    tail_history = []

    saturation_iteration = None

    for iteration in range(MAX_ITER + 1):
        # ---- Diagnostics ----
        tail = zipf_tail_mass(current_tokens)
        tail_delta = tail - base_tail_mass
        risk_state = classify_risk("tail_mass_delta", tail_delta)

        history.append({
            "iteration": iteration,
            "tail_mass": tail,
            "tail_mass_delta": tail_delta,
            "risk_state": risk_state,
        })

        print(f"[D{iteration}] tail_mass={tail:.6f} → {risk_state}")

        # ---- Saturation check (skip for first few iterations) ----
        tail_history.append(tail)
        if len(tail_history) >= SAT_WINDOW:
            recent = tail_history[-SAT_WINDOW:]
            if max(recent) - min(recent) < SAT_EPS:
                saturation_iteration = iteration
                print(f"⚠️  Saturation detected at iteration {iteration}")
                break

        # ---- Stop if max iteration reached ----
        if iteration == MAX_ITER:
            break

        # ---- Self-training step (warm-start, model persists) ----
        n_original = int(ALPHA * len(original_tokens))
        n_synthetic = len(original_tokens) - n_original
        synthetic = generate_tokens(
            model=model,
            seed_tokens=None,
            sample_size=n_synthetic,
            rng=rng,
        )

        current_tokens = mix_tokens(
            original_tokens=original_tokens,
            synthetic_tokens=synthetic,
            alpha=ALPHA,
        )

        model.train(current_tokens)

    # ------------------
    # SAVE RESULTS
    # ------------------

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "alpha": ALPHA,
        "metric": "tail_mass",
        "max_iterations": MAX_ITER,
        "saturation_iteration": saturation_iteration,
        "iterations": history,
    }

    OUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nSaved Phase 6A results to {OUT_PATH}")


if __name__ == "__main__":
    main()
