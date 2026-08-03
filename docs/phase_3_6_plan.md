# Phase 3.6 — Explicit Representational Bottleneck

## Motivation
Phase 3.5 shows that self-training degradation is reversible
when information is preserved.

We now test whether irreversibility emerges only when the system
explicitly destroys information.

## Bottleneck Definition
A fixed vocabulary constraint is introduced after initial training.
Tokens outside the frozen vocabulary are mapped to UNK.

This constraint is:
- Explicit
- Localized
- Causally isolated
- Independent of model architecture

## Hypothesis
Under a representational bottleneck, recovery on clean data will fail
to restore the original distribution beyond a measurable threshold.

## Constraints
- Same model
- Same training protocol
- Same data
- One bottleneck only

## Success Criteria
Recovered distribution remains statistically distinct from original
after retraining.


## Note (flagged, not resolved)

Verified directly this session against the actual code: the implemented
bottleneck mechanism does not match this document's description.

This document describes a fixed vocabulary constraint with out of
vocabulary tokens mapped to UNK. The actual implementation,
src/models/constraints.py's apply_frequency_cutoff function, does
something different: it drops any token with frequency below min_count
entirely from the counter, with no UNK substitution and no fixed
vocabulary defined in advance. It is a frequency based cutoff, not a
vocabulary freeze with UNK mapping.

Every usage seen throughout this session, run_phase3_6, run_phase5
scripts, and elsewhere, invokes this via model_config equals
min_token_count colon 3, confirming apply_frequency_cutoff is the actual,
sole constraint mechanism in use, not a UNK based vocabulary freeze.

Not resolved here. Worth reconciling whether this document describes an
earlier planned design that was later implemented differently, or
whether this is simply a drift between plan and implementation that was
never noticed.

## Note (confirmed, traced end to end)

Follow-up to the note above. Traced the actual mechanism precisely this
session: run_phase36.py and run_recovery_threshold.py pass
model_config={"min_token_count": 3} into model construction.
model_factory.py forwards this into the n-gram model's constructor.
src/models/ngram.py stores it and calls apply_frequency_cutoff from
src/models/constraints.py during training, permanently dropping any
token below the threshold from the count table.

Separately, src/data/bottlenecks.py exists and implements exactly what
this document's main text describes: freeze_vocabulary builds a fixed
vocabulary from D0, apply_vocab_bottleneck maps out-of-vocabulary tokens
to <UNK>. Confirmed by a full-codebase search that neither function in
this file is imported or called anywhere. This file is genuinely dead
code, not a stale duplicate of the live mechanism, a fully unused,
never-wired-in implementation of the originally planned design.

So the drift is now precisely characterized: this document's main text
describes bottlenecks.py's mechanism (vocabulary freeze plus UNK
substitution), but every experiment that runs actually uses
constraints.py's mechanism (frequency cutoff, permanent deletion, no
UNK token). These are meaningfully different: frequency cutoff removes
rare tokens from the model's representation entirely at every
retraining, while vocabulary freeze plus UNK would preserve rare tokens
as a collapsed shared signal rather than deleting them outright.

Not resolved here. The remaining decision is editorial, not diagnostic:
either delete src/data/bottlenecks.py since it is unused, or keep it
and label it clearly as an unused alternate design rather than
implying, by its presence and naming, that it is part of the working
pipeline.