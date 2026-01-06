# PHASE 3.6 — EXPLICIT REPRESENTATIONAL BOTTLENECK
# Vocabulary freeze + UNK mapping

from collections import Counter


def freeze_vocabulary(tokens, max_vocab_size=None):
    """
    Freeze vocabulary from initial dataset (D0).

    If max_vocab_size is set, keep only top-K most frequent tokens.
    """
    counts = Counter(tokens)
    if max_vocab_size is not None:
        vocab = set(tok for tok, _ in counts.most_common(max_vocab_size))
    else:
        vocab = set(counts.keys())
    return vocab


def apply_vocab_bottleneck(tokens, vocab, unk_token="<UNK>"):
    """
    Map out-of-vocabulary tokens to UNK.
    """
    return [tok if tok in vocab else unk_token for tok in tokens]
