# PHASE 3 LOCKED
# Causal self-training loop.
# Do not modify without invalidating downstream claims.

def mix_tokens(original_tokens, synthetic_tokens, alpha):
    """
    Construct D_k+1 = alpha * D0 + (1 - alpha) * S_k
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    n_original = int(alpha * len(original_tokens))
    n_synthetic = len(original_tokens) - n_original

    if len(synthetic_tokens) < n_synthetic:
        raise ValueError(
            f"mix_tokens requires {n_synthetic} synthetic tokens to honor "
            f"alpha={alpha} against a {len(original_tokens)}-token corpus, "
            f"but only {len(synthetic_tokens)} were provided. Silently "
            f"truncating would corrupt the alpha-mixture and understate "
            f"contamination -- see docs/phase_3_mixture_fix.md."
        )

    return (
        original_tokens[:n_original] +
        synthetic_tokens[:n_synthetic]
    )