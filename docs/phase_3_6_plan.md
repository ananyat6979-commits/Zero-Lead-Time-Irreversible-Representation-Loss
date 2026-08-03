# Phase 3.6: Explicit Representational Bottleneck

## Motivation
Phase 3.5 shows that self-training degradation is reversible
when information is preserved.

We now test whether irreversibility emerges only when the system
explicitly destroys information.

## Bottleneck Definition
A frequency-based representational constraint is applied at every
training call: any token whose count falls below `min_token_count`
(here, 3) is dropped entirely from the model's distribution before
sampling. This is implemented in `src/models/constraints.py`'s
`apply_frequency_cutoff`, invoked via `model_config={"min_token_count": 3}`.

Note this differs from an earlier draft of this plan, which described
a fixed-vocabulary constraint (freeze the vocabulary from D0, map
out-of-vocabulary tokens to UNK). That mechanism was never implemented
for Phase 3.6; it exists separately as `src/data/bottlenecks.py`
(`freeze_vocabulary` / `apply_vocab_bottleneck`), which is not called
by this phase or any other. The two mechanisms are meaningfully
different: a frequency cutoff re-evaluates the eligible token set at
every iteration based on current corpus composition (dynamic), while a
vocabulary freeze fixes the eligible set once at D0 and never revisits
it (static). Phase 3.6's actual result reflects the dynamic,
frequency-cutoff mechanism only. Testing the static vocabulary-freeze
variant is a legitimate, separate follow-up experiment, not yet run.

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


## Resolution note (previously flagged, now resolved)

An earlier version of this document described a fixed-vocabulary/UNK
mechanism that didn't match the actual implementation. The Bottleneck
Definition section above has been corrected to describe the real
mechanism (frequency cutoff via `apply_frequency_cutoff`). The unused
`src/data/bottlenecks.py` file, which implemented the originally
planned but never-adopted vocabulary-freeze mechanism, has been removed
from the active codebase: see project history for the original
implementation if the static-vocabulary variant is revisited later.