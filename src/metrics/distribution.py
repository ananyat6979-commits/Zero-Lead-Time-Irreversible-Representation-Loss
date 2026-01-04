"""
Distribution-level diagnostics.

All metrics operate on empirical token or n-gram distributions.
No model internals are used.
"""

from collections import Counter
import math
import numpy as np

def empirical_distribution(tokens):
    """
    Returns a normalized empirical distribution over tokens.
    """
    counts = Counter(tokens)
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}

def shannon_entropy(dist):
    """
    Computes Shannon entropy H(P) = -Σ p log2 p
    """
    entropy = 0.0
    for p in dist.values():
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def kl_divergence(p, q, eps=1e-12):
    """
    KL(P || Q)
    """
    kl = 0.0
    for key, p_val in p.items():
        q_val = q.get(key, eps)
        kl += p_val * math.log2(p_val / q_val)
    return kl

def js_divergence(p, q):
    """
    Jensen-Shannon divergence between P and Q.
    Symmetric and bounded.
    """
    m = {}
    for key in set(p) | set(q):
        m[key] = 0.5 * p.get(key, 0) + 0.5 * q.get(key, 0)

    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def type_token_ratio(tokens):
    """
    |V| / N
    """
    return len(set(tokens)) / len(tokens)

def zipf_tail_mass(tokens, tail_fraction=0.1):
    """
    Fraction of probability mass in the lowest-frequency tail.
    """
    counts = Counter(tokens)
    total = sum(counts.values())

    sorted_items = sorted(counts.items(), key=lambda x: x[1])
    cutoff = int(len(sorted_items) * tail_fraction)

    tail_items = sorted_items[:cutoff]
    tail_mass = sum(count for _, count in tail_items) / total

    return tail_mass
