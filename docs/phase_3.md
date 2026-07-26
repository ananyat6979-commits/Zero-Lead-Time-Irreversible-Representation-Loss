# Phase 3 — Causal Self-Training Loop

## Purpose
Establish a causally clean, deterministic self-training loop
to study distributional degradation under controlled contamination.

## Scope (Locked)

- Model: UnigramLM only
- Sampling: deterministic under injected RNG
- Data: fixed original corpus D₀
- Update rule: Dₖ₊₁ = α·D₀ + (1−α)·Sₖ
- Training: model retrained from scratch each iteration

## Explicit Exclusions

- Context-dependent models (e.g. bigram, trigram)
- Metrics, diagnostics, or early stopping
- Any randomness outside injected RNG

Phase 3 restricts to UnigramLM to eliminate context-dependent generation.

## Status
Frozen. Any modification invalidates downstream experiments.
