# Addendum c to PREREG_cb_drive_speed.md (written 2026-10-12, after steps 1–6; earlier files unchanged)

Same frozen metric (`cb_drive_speed.py`, SHA-256 `009e06e2…`). Registered before any 2026 data.

- **C1 (value).** In 2026 targeted CB coverage reps, completion over expected allowed declines with drive speed.
  - xComp model = the logistic model of `src/62_cb_value.py`, refitted on 2026 pool-receiver targets with the same
    pre-throw features.
  - Test: one-sided clustered p < .05. In 2023–25: −7.2 pp per SD, permutation p .005.
- **C2 (no avoidance; reported regardless).** Targets per coverage rep do not depend on drive speed. In 2023–25:
  −0.7 pp per SD, p .42.
- **C3 (zone).** In a crossed random-effects model (CB, receiver) of separation gained at the break, the
  drive × zone interaction is negative. In 2023–25: −0.118, p .026.
- **C4 (projection).** Forecasting each 2026 CB's separation allowed at the throw (context-adjusted, his 2026 reps)
  with the combine-only forecaster beats the league average.
  - The forecaster is b × z(drive). b is fitted on the 2023–25 reps only.
  - Test: weighted MSE improvement > 0, on CBs with ≥ 10 reps. In 2023–25 (leave-one-class-out): +14% to +23%.
- **Negative results registered as such (expected to stay null).**
  - The physics sprint-curve parameters (asymptotic drive speed, time constant, transition tax) do not beat the
    frozen drive top speed.
  - Speed carried through the turn does not predict man coverage.
  - The receiver-free in-game "transition drive" does not bridge to the combine.
