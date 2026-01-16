"""
Simple n-gram language models.

These models are intentionally naive.
Transparency > performance.
"""

# NOTE:
# Explicit start/end tokens are intentionally omitted at this stage
# to avoid introducing boundary artifacts during early collapse analysis.

from collections import Counter
import math
from src.models.constraints import apply_frequency_cutoff


def tokenize(text):
    return text.strip().split()


class UnigramLM:
    def __init__(self, min_token_count=None):
        self.counts = Counter()
        self.total = 0
        self.min_token_count = min_token_count


    def train(self, tokens):
        self.counts.update(tokens)

        if self.min_token_count is not None:
            self.counts = apply_frequency_cutoff(
                self.counts,
                self.min_token_count
            )

        self.total = sum(self.counts.values())
    
        if self.min_token_count is not None:
            assert all(
                count >= self.min_token_count
                for count in self.counts.values()
    )


    
    def prob(self, tok):
        vocab_size = len(self.counts)
        return (self.counts.get(tok, 0) + 1) / (self.total + vocab_size)


    def sample(self, sample_size, rng):
        tokens = list(self.counts.keys())
        probs = [self.prob(t) for t in tokens]
        return rng.choices(tokens, probs, k=sample_size)
    

    def cross_entropy(self, tokens):
        entropy = 0.0
        for tok in tokens:
            p = self.prob(tok)
            entropy -= math.log2(p)
        return entropy / len(tokens)
    


class BigramLM:
    def __init__(self):
        self.counts = {}
        self.context_totals = {}
        self.vocab = set()

    def train(self, tokens):
        # Vocabulary is derived from training tokens only.
        # No external vocabulary expansion is used.
        self.vocab = set(tokens)

        for i in range(len(tokens) - 1):
            ctx = tokens[i]
            nxt = tokens[i + 1]
            self.counts.setdefault(ctx, Counter())[nxt] += 1
            self.context_totals[ctx] = self.context_totals.get(ctx, 0) + 1

    def prob(self, ctx, tok):
        # Unseen tokens are handled via add-one smoothing.
        # No explicit OOV token is used.
        return (
            self.counts.get(ctx, {}).get(tok, 0) + 1
        ) / (
            self.context_totals.get(ctx, 0) + len(self.vocab)
        )

    def sample(self, start_token, sample_size, rng):
        tokens = [start_token]

        for _ in range(sample_size - 1):
            ctx = tokens[-1]
            # Sampling considers full vocabulary.
            # This increases entropy artificially but preserves support visibility.
            candidates = list(self.vocab)
            probs = [self.prob(ctx, t) for t in candidates]
            tokens.append(rng.choices(candidates, probs)[0])

        return tokens

    def cross_entropy(self, tokens):
        entropy = 0.0
        count = 0

        for i in range(len(tokens) - 1):
            ctx = tokens[i]
            nxt = tokens[i + 1]
            p = self.prob(ctx, nxt)
            entropy -= math.log2(p)
            count += 1

        return entropy / count


class TrigramLM:
    def __init__(self):
        # maps (w1, w2) -> Counter of next words
        self.counts = {}
        # maps (w1, w2) -> total count
        self.context_totals = {}
        # set of all tokens
        self.vocab = set()

    def train(self, tokens):
        # Vocabulary is derived from training tokens only.
        # No external vocabulary expansion is used.
        self.vocab = set(tokens)

        for i in range(len(tokens) - 2):
            ctx = (tokens[i], tokens[i + 1])
            nxt = tokens[i + 2]

            self.counts.setdefault(ctx, Counter())[nxt] += 1
            self.context_totals[ctx] = self.context_totals.get(ctx, 0) + 1

    def prob(self, ctx, tok):
        # Unseen tokens are handled via add-one smoothing.
        # No explicit OOV token is used.
        return (
            self.counts.get(ctx, {}).get(tok, 0) + 1
        ) / (
            self.context_totals.get(ctx, 0) + len(self.vocab)
        )

    def sample(self, start_tokens, sample_size, rng):
        """
        start_tokens must be a tuple of two tokens
        """
        tokens = list(start_tokens)

        for _ in range(sample_size - 2):
            ctx = (tokens[-2], tokens[-1])
            # Sampling considers full vocabulary.
            # This increases entropy artificially but preserves support visibility.
            candidates = list(self.vocab)
            probs = [self.prob(ctx, t) for t in candidates]
            tokens.append(rng.choices(candidates, probs)[0])

        return tokens

    def cross_entropy(self, tokens):
        entropy = 0.0

        for i in range(len(tokens) - 2):
            ctx = (tokens[i], tokens[i + 1])
            nxt = tokens[i + 2]
            p = self.prob(ctx, nxt)
            entropy -= math.log2(p)

        return entropy / (len(tokens) - 2)
