# Phase 3.5 — Recovery Counterfactual Results

## Objective
Test whether distributional degradation induced by self-training
is reversible when retraining on clean data under identical compute
and model constraints.

## Experimental Setup
- Model: UnigramLM
- Corpus: Project Gutenberg (Pride and Prejudice)
- Training protocol: identical to Phase 3
- No synthetic data during recovery
- Deterministic execution

## Observations
Across contamination depths k ∈ {3,5,7,10}:

- Recovered distributions converge to original
- JS(original || recovered) ≈ 0
- Entropy restored
- Zipf tail mass restored
- Type-token ratio restored

## Interpretation
Self-training alone does not induce irreversible distributional collapse.
Observed degradation is reversible when information is preserved.

## Implication
Irreversibility, if present, must arise from an explicit information-destroying
constraint, not from self-training dynamics alone.

## Note (flagged, not resolved)

Verified directly this session against the actual codebase: neither
run_recovery_control.py nor run_recovery_threshold.py currently produce
a JS divergence approximately zero for recovered distributions.

run_recovery_control.py's js_recovered was confirmed to be
0.027955425229532538, and this value was also confirmed to be invariant
to contamination entirely, since retrain_from_contaminated defaults to
false there, meaning it measures the pristine sampling floor, not
recovery from any specific contamination depth.

run_recovery_threshold.py, using the corrected retrain_from_contaminated
equals true path added this session, produced js_recovered values around
0.045 to 0.049 across k from 1 to 10, also not approximately zero.

This document's claim of JS approximately zero may describe an earlier,
different, or idealized version of these results, not the current state
of the code. Not resolved here, worth reconciling before treating this
document's claims as current.
