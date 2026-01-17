# Zero-Lead-Time Irreversible Representation Loss (ZL-IRL)

## Trigger
A deterministic representation constraint applied before or during self-training that irreversibly truncates representational support.

## Observable Signature
- Immediate divergence from the original data distribution
- No statistically detectable precursor under standard distributional metrics
- Negative controls remain flat under identical conditions

## Consequence
- Recovery converges to a degraded attractor
- Retraining on clean data fails to restore representational support
- Monitoring-based intervention is ineffective after regime entry
