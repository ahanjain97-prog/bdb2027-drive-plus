# Pre-registration: CB Transition Drive Speed → coverage at the break (frozen 2026-10-12)

This file was written before any 2026-season game data or 2026-class combine data was seen. Nothing below may be
changed after that data arrives. If it ever is changed, the change must be dated and described here.

## Frozen artifacts
| file | SHA-256 |
|---|---|
| `cb_drive_speed.py` (metric definition) | `009e06e26a218db0b99a6df24f8c93711532a2d28aa28be2d80f1f4f8d8d3ccb` |
| `cb_drive_speed_frozen_2026-10-12.csv` (values for the 121 DBs with the drill, 2023–25 classes) | `634afeae66b9ec97b8ec584c09b87d099148f20eaf43a7598feb46e8eacbc5c0` |

The metric is the top speed (yd/s) a defensive back reaches when he drives out of the ~45° transition in the
BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION drill. The transition is located by heading change, not by a speed dip:
good corners keep their speed through the turn.

## How it was found (2023–25 seasons; cannot be reused as the test)
It is an exploratory finding. The drill's top speed was picked by the open, selection-aware screen (tier B) for "CB
separation allowed at the break", with the same sign in all three held-out draft classes (−.28, −.36, −.21). Then it
was checked at the rep level in 1,707 CB coverage reps (a pool WR/TE vs a pool CB, both tracked).

| check | result |
|---|---|
| separation gained at the receiver's break, per SD | −0.079 yd (two-way clustered p .007; player permutation p .002) |
| + tracked 40 top speed + weight + height | −0.074 yd (p < .001) |
| official 40 time instead of drive speed | −0.007 (p .87) |
| by draft class | 2023 −.125 · 2024 −.069 · 2025 −.062 |
| leave-one-CB-out range | −.066 to −.100 |
| separation at the throw | −0.106 yd per SD (permutation p .047) |
| completion rate when targeted | −10.3 pp per SD (p < .001, 284 targets) |
| by coverage | zone −0.127 (p .055) vs man −0.033 (p .16) |

Drill phases that do not carry the effect: backpedal speed, speed kept through the turn, turn time, first-second
drive acceleration.

## Hypotheses for 2026 (directional, one-sided)
- **H1 (primary).** In 2026 CB coverage reps, CB Transition Drive Speed (z-scored over the CBs in the reps) has a
  negative coefficient on separation gained at the receiver's break. Model: + the receiver's separation skill in his
  other reps + route + man/zone, two-way clustered. Test: player-permutation p < .05 (one-sided).
- **H2.** Same model with completion when targeted as the outcome: coefficient < 0, one-sided clustered p < .05.
- **H3.** H1 holds after adding tracked 40 top speed, weight and height.
- **H4 (exploratory, reported regardless).** The coefficient is more negative in zone than in man coverage.

## Outcome and rep definitions (fixed)
Exactly as in `src/41_coverage_reps.py` and `src/56_cb_transition.py` as of the date above:
- A coverage rep = mean receiver–CB distance ≤ 3 yd over ≥ 1 s of the route (0.5 s after the snap until the throw,
  max 3.5 s).
- The receiver's break = his peak smoothed lateral acceleration at > 2 yd/s between 0.5 and 3 s.
- Separation gained at the break = distance 0.5 s after the break minus distance 0.3 s before it.

## Samples
- 2026 rookies: compute the metric with the frozen file.
- Returning CBs (2023–25 classes): use the frozen values above.

Pooled is primary. Minimum: 25 CBs and 600 reps; below that, report as "inconclusive".

## Decision
Claim "drive speed out of the transition predicts coverage at the break" only if H1 and H3 pass. If H1 fails, the
metric is retired.
