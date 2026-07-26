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