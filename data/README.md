# Data Directory

This project enforces strict data lineage.

## raw/
Human-authored, frozen corpora.
Never modified after ingestion.

## generated/
Model-generated synthetic data.
Each file corresponds to a specific model iteration.

## mixtures/
Explicit mixtures of raw and generated data.
Mixture ratios are logged and reproducible.

No data file may exist without a corresponding lineage record.