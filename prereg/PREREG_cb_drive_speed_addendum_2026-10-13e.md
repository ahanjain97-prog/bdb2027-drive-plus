# Addendum e to PREREG_cb_drive_speed.md: wide sample and evidence standard (written 2026-10-13; earlier files unchanged)

Written before any 2026-season game data or 2026-class combine data was seen. Nothing here may change once that data
arrives. The frozen metric (`drive_plus.py`, SHA-256 `f1a2847b…`) and the primary hypotheses D1 and D3 of addendum d
are unchanged; this addendum adds a registered secondary sample and states the evidence standard in advance, because
the available data are thin (about 50–80 defensive backs) and the standard must be fixed before, not after, 2026.

## Why
The audit of 2026-10-13 (`code/88_audit.py`, SHA-256 `237544e09d4e349fca495a92cc8e00a5d92b843ab2cfb91b48f3ded765aa995f`)
found that the registered rep definition selects on a window that includes the outcome (mean separation ≤ 3 yd over
the whole route), which attenuated the effects and made the separation claims depend on vertical routes. A
pre-break-only definition removes that selection.

## Registered secondary sample: the "wide" reps
Exactly as built in `code/89_wide_validation.py` (SHA-256 `9244783b7714dd485802bd29ebb0cd514cd8aa1af69c7409c0c1bbfc5d262518`):
- pool receiver (WR, TE, RB, FB) running a route and a graded DB on the same dropback, REG + POST, not nullified;
- kept when the mean receiver–DB distance from 0.5 s after the snap **to the receiver's key move** is ≤ 4 yd, with
  ≥ 4 pre-move frames; the key move = the receiver's peak smoothed lateral acceleration at > 2 yd/s, 0.5–3 s;
- zone-matching = zone call and pre-move velocity alignment ≥ .5 (the registered cell at ≥ .7 is also reported);
- all graded DBs with a position control; corners-only reported alongside.
In 2023–25 this gives 2,952 reps, 76 DBs, 502 targets. Radius 3 and 5 yd gave the same conclusions.

## Evidence standard (fixed now, applied to every claim in both directions)
Hypotheses are directional, so tests are one-sided.
- **Confirmed:** one-sided player-permutation p < .05.
- **Supported:** one-sided p < .10 **and** the 80% interval (clustered SE) excludes zero **and** the expected sign.
- **Right sign only / wrong sign** otherwise.
The same tiers apply to the registered nulls (team-level EPA, box-score breakups, head-around, other drills). A null
that reaches "supported" under this standard is reported as such; a hypothesis that fails it is reported as failed.
Sign consistency across 2024-type and 2025-type seasons is reported for every claim.

## 2026 tests on the wide sample (secondary to D1–D3; all expected signs as in 2023–25)
- W1 separation gained at the key move (−); W2 closing after it in zone-matching reps (+) and in all zone reps (+);
  W3 receiver open at the throw, ≥ 2 yd (−); W4 separation at the throw (−); W5 completion when targeted (−).
- Reported splits: corners only; alignment ≥ .7; without GO routes; each season separately.
- 2023–25 reference values (per 10 Drive+ points, 80% intervals): W1 −0.16 [−0.24, −0.08]; W2 +0.25 [+0.12, +0.38]
  zone-matching and +0.38 [+0.20, +0.57] all zone; W3 −4.6 pts [−6.3, −3.0]; W4 −0.36 yd [−0.52, −0.21];
  W5 −8.5 pts [−11.1, −5.8]. All reached "supported" in every split except completion in the 2024 season alone
  (−2.9 pts, right sign only).

## Decision (unchanged in structure)
D1 and D3 remain the primaries at one-sided p < .05. The wide-sample tests W1–W5 are reported in full under the tiers
above and may raise a "partial" result to "supported" but not to "confirmed".
