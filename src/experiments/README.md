# Experimental Provenance — ZL-IRL

This document records the **complete experimental path** of the project:
all hypotheses tested, mechanisms isolated, negative results preserved, and conclusions locked.

It exists to prevent overclaiming.

This repository does not hide failed ideas.
It preserves them.

---

## How to Read This Document

This document corresponds to the experimental and methodological sections of a conventional ML paper.

The root README states the final boundary result.
This document provides the complete experimental path supporting it.

---

## 1. Motivation & Problem Framing

Modern ML systems increasingly retrain on partially self-generated data.

In production systems, retraining decisions are often driven by:

- performance metrics (accuracy, perplexity)
- user-visible failures
- downstream incidents

These signals are lagging indicators.

By the time they move, systems may already be operating in a degraded or risky regime.

The central question motivating this project was:

> **Can distributional diagnostics detect harmful self-training dynamics early — before surface metrics degrade and before recovery becomes difficult or impossible?**

Crucially, we did not assume:

- that collapse is inevitable  
- that irreversibility always occurs  
- that degradation must lead to failure  

Those assumptions were tested, not baked in.

---

## 2. Experimental Philosophy (Systems, Not Models)

This project studies **self-training pipelines as systems**, not as models.

Across all phases:

- model class is fixed (memoryless n-gram LMs)
- retraining succeeds at every iteration
- original human data is partially reintroduced
- no architectural escalation is used
- RNG and training semantics are fixed and controlled
- each phase isolates exactly one causal factor

The goal is causal isolation, not performance optimization.

Why n-gram models?

N-gram language models are used deliberately as a minimal, fully inspectable instantiation of a self-training system.

They serve as a lower-bound witness: if irreversible, zero-lead-time failure occurs in a memoryless model with no hidden state, gradient dynamics, or optimization instability, then the phenomenon cannot be attributed to expressivity, depth, or training instability.

The contribution is about system-level causality, not model capacity.

---

## 3. Phase Overview

| Phase | Question | Outcome |
|-----|--------|--------|
| Phase 3 | Does self-training alone induce degradation? | Yes (reversible) |
| Phase 3.5 | Can reset retraining recover? | Yes |
| Phase 5 | Are early distributional diagnostics informative? | Yes |
| Phase 3.6 | Do representation constraints induce irreversibility? | Yes (boundary) |
| Phase 4 | Does state persistence alone cause irreversibility? | No |
| Phase 5B | Are all diagnostics equally useful? | No |

---

## 4. Phase 3 — Self-Generated Feedback (Baseline)

**Question**  
Does self-training alone induce irreversible degradation?

**Setup**

A conservative self-training loop where:

- model class is fixed
- training succeeds at every iteration
- original human data is partially reintroduced
- no architectural or preprocessing changes are made

**Findings (Locked)**

After a single self-training iteration:

- Jensen–Shannon divergence from the original data jumps sharply
- entropy decreases measurably
- rare-token mass increases
- type–token ratio shifts

All of this occurs **before**:

- visible quality degradation
- instability
- irreversibility
- recovery failure

There is no gradual lead-in.
The regime shift happens at entry.

Across subsequent iterations:

- metrics do not drift monotonically
- variance is low
- the system stabilizes into a degraded equilibrium

**Conclusion**

Self-training alone induces **silent but reversible distributional degradation**.

This establishes the reversible regime.

---

## 5. Phase 3.5 — Recovery Under Reset Retraining

After contamination, models are retrained **from scratch** on clean data.

**Finding (Locked)**

- Reset retraining reliably recovers the original distribution
- No irreversibility is observed
- Distributional support is restored

This is a **negative result**, preserved intentionally.

Recovery is not inherently impossible.

---

## 6. Phase 5 — Early-Warning Diagnostics

We evaluate distributional diagnostics across self-training iterations:

- Jensen–Shannon divergence to D₀
- Shannon entropy
- Zipf tail mass
- Type–token ratio

**Observed Pattern**

- D₀ → D₁: abrupt regime shift across all metrics
- D₁ → D₁₀: stable plateau with minimal variance

There is no gradual approach.
The regime shift happens at entry.

These diagnostics:

- move before surface metrics
- remain informative even when recovery is still possible
- do not depend on collapse or irreversibility

They are **early-warning signals**, not post-mortem indicators.

---

## 7. Phase 5B — Diagnostic Robustness

Not all diagnostics behave equally post-trigger.

**Finding (Locked)**

- Tail mass and type–token ratio degrade monotonically once triggered
- Entropy and JS divergence fluctuate post-trigger
- Tail-sensitive metrics provide the most reliable early-warning signal

This discriminates *useful* diagnostics from merely reactive ones.

---

## 8. Phase 3.6 — Explicit Representation Constraint (Boundary Case)

We introduce a single change:

> a deterministic representation constraint applied inside the model

All other factors remain identical.

**Findings (Locked)**

- Distributional divergence occurs immediately
- Recovery retraining fails to restore representational support
- Loss of tail mass persists
- No precursor signal exists

The system transitions directly from **SAFE → HIGH_RISK**.

This is a boundary result.

---

## 9. Phase 4 — State Persistence (Warm-Start Recovery)

We test whether training history alone induces irreversibility.

**Setup**

- recovery warm-started from contaminated parameters
- data, model class, and metrics unchanged

**Finding (Locked)**

- warm-start recovery degrades relative to reset retraining
- no sharp irreversibility threshold observed
- recovery quality varies smoothly with contamination level

This demonstrates **conditional path dependence without collapse**.

---

## 10. Synthesis — From Degradation to Boundary Failure

Across all phases:

- degradation can occur without irreversibility
- recovery is often possible
- early diagnostics detect regime entry reliably
- irreversibility is not guaranteed

Irreversibility emerges **only** when deterministic representation truncation is introduced.

This motivates the definition of ZL-IRL.

---

## 11. Boundary Result — Zero-Lead-Time IRL

**Trigger**

A deterministic representation constraint that irreversibly truncates representational support.

**Observable Signature**

- immediate regime shift
- no detectable precursor signal
- flat negative controls under identical conditions

**Consequence**

- recovery converges to a degraded attractor
- clean retraining fails
- monitoring cannot intervene post-entry

Early warning is structurally impossible.

---

## 12. What Failed to Induce Irreversibility

The following mechanisms **did not** induce irreversibility:

- self-training alone
- reset retraining
- warm-start without truncation
- smooth contamination

Negative results matter.
They narrow the hypothesis space.

---

## 13. Limitations (Explicit)

- model class: simple n-gram models
- tokenization: whitespace
- corpus: single public-domain text
- no performance metrics (deliberate)

These are design choices, not omissions.

---

## 14. Final Locked Conclusions

Silent distributional degradation can emerge under partial self-training even when training succeeds.

Conventional performance metrics are insufficient early indicators.

Recovery is possible under reset retraining.

Irreversibility is conditional and rare.

When it occurs via deterministic representation truncation, it exhibits **zero lead time**.

This project proves the null case, isolates causality, preserves negative results, and tightens claims instead of inflating them.

That is the contribution.