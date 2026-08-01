### Figure X — Zero-lead-time regime entry under representation constraint

JS divergence to the original data distribution (P₀) across self-training iterations.
The dashed line indicates the maximum deviation observed under fully reversible self-training (Phase 3.5).

Under a deterministic representation constraint, the system transitions immediately into a post-entry regime at the first self-training iteration, with no observable approach toward the boundary.
Subsequent iterations remain stably separated from the reversible envelope.

This figure does not, by itself, establish irreversibility.
Irreversibility is established by the failure of recovery experiments beyond the entry point.

Figure 1. After self-training under a deterministic representation constraint, retraining on clean data converges to a higher-divergence fixed point, demonstrating irreversible representational loss.

## Figure: Canonical ZL-IRL Boundary Result

Jensen–Shannon divergence to the original distribution across self-training iterations.

The reversible regime (Phase 3.5) remains below the threshold derived from recovery.
Introducing deterministic representation truncation (Phase 3.6) causes an immediate, irreversible divergence with no detectable precursor.

## Note (flagged, not resolved)

Verified directly this session against the actual code: the threshold
calibration in run_phase5_calibrate_thresholds.py had a sign error,
mu + K*sd instead of mu - K*sd, which made the HIGH_RISK threshold
numerically less extreme than WARNING despite K_HIGH=3.0 > K_WARNING=1.0.
Fixed by correcting the sign.

Rerunning the three scripts that consume these thresholds after the fix:

- run_phase5_warmstart_alerts.py (alpha=0.1): now SAFE through
  iteration 14, WARNING only at iteration 15, never reaches HIGH_RISK.
  Previously HIGH_RISK from iteration 11.
- run_phase6_iteration_boundary.py (alpha=0.5): HIGH_RISK from
  iteration 2 onward, consistent. Previously iteration 1.
- run_phase5_alerts.py (alpha=0.8): SAFE through iteration 4, WARNING
  from iteration 5 onward, never reaches HIGH_RISK. Previously
  HIGH_RISK from iteration 3.

This does not by itself confirm or refute this document's "immediate,
no detectable precursor" claim, that depends on the canonical figure's
underlying JS-divergence curve, which has not been reverified this
session. But it does mean any claim of "immediate HIGH_RISK
classification" specifically should be treated as resting on the old,
miscalibrated thresholds, and needs to be re-examined against the
corrected ones before being restated as current.

Not resolved here. Worth reconciling once the canonical figure itself
has been reverified.