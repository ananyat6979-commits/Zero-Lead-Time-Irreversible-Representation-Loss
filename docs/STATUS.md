## Project Status — Scope Locked

This repository studies a single failure mode:

**Zero-Lead-Time Irreversible Representation Loss (ZL-IRL)**

Definition:
A learning-system failure where deterministic representation constraints cause
instantaneous, unrecoverable loss of task-relevant distributional support.

Properties:
- Failure enters the system at constraint application time
- No precursor signal exists in post-constraint representations
- Retraining from clean data does not recover lost support

Out of scope:
- Monitoring systems
- Production early-warning tools
- Latent-space observability
- Gradual degradation regimes

Any future work on detection or monitoring must live in a separate repository.
