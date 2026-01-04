## Variables
- model_type: unigram | bigram | trigram
- alpha: [1.0, 0.8, 0.6, 0.4, 0.2]
- iterations: fixed (e.g. 10)
- sample_size: fixed
- seed: fixed

## Experiments
1. Forward self-training (contamination)
2. Recovery run (clean retraining)
3. Clean baseline (no contamination)

## Constraints
- Same model
- Same data size
- Same iterations
- No hyperparameter changes
