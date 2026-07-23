from collections import Counter

def apply_frequency_cutoff(counts: Counter, min_count: int) -> Counter:
    """
    Drops tokens with frequency < min_count.
    Enforces irreversible representation loss.
    """
    return Counter({
        token: count
        for token, count in counts.items()
        if count >= min_count
    })
