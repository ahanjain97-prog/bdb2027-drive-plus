# Addendum d to PREREG_cb_drive_speed.md: Drive+ (written 2026-10-12; earlier files unchanged)

Written after the 2023–25 analyses below and before any 2026-season game data or 2026-class combine data was seen.
Nothing here may change once that data arrives; any change must be dated and described in this file.

## Frozen artifacts
| file | SHA-256 |
|---|---|
| `drive_plus.py` (the grade) | `f1a2847b248efce018aaeb884cf0e8ffdde2aadee2dcb098b48a2ced66f55bc6` |
| `drive_plus_frozen_2026-10-12.csv` (121 graded players, 2023–25 classes) | `2c60e8dbd0489860dc4ff417bfbd7d0dfed4a4407f89cc8229f3c08124ad1f7c` |
| `cb_drive_speed.py` (the underlying metric, unchanged) | `009e06e26a218db0b99a6df24f8c93711532a2d28aa28be2d80f1f4f8d8d3ccb` |

Outcome and rep definitions are fixed as the code in these files on this date:

| definition | file | SHA-256 |
|---|---|---|
| coverage reps, receiver's break, separation gained | `code/41_coverage_reps.py` | `61a29037f7773299f71621b2c5f3890dc54576b5109d59dd553e148b3cc27ec0` |
| closing after the break (0.7 s) | `code/57_cb_mechanism.py` | `b0eb88c2256d7248304c2e92f44e5664842238512e939dc77775fdffdb3e9cab` |
| zone-matching cell (zone + pre-break alignment ≥ .7) | `code/70_cb_match_zone.py` | `312c9463f1151d90b3ae89b9b482e63789c1b35eb9321d30ff17a1d3091060c7` |
| speed 1.0 s after the turn (v_10) | `code/72a_drive_decomposition.py` | `51d2e1a934967e7fea7f107ca67985f9a5d782b4c911a77b6345b575342e0fba` |
| separation time course by tercile | `code/73g_separation_curve.py` | `16b875ad61fb17e515af3b943c2c609b7b789e2852a7c95ec93b4750d474dfb1` |
| throws toward vs away | `code/76b_toward_away.py` | `121aa3f7907784dcdf8c13e3ed456a07d466e137672db99630cc47e5233afb33` |

## What Drive+ is
`Drive+ = 100 + 10 × (z − 0.0794) / 1.0405`. Here `z` is the frozen drive speed, z-scored within the combine class
over every player with a valid value. The constants put the 2023–25 pool CBs at mean 100 and SD 10.
Tiers are 110+, 100–110, 90–100 and <90.

**Why one drill.** In 2023–25 the single drill beat every multi-drill version out of sample:
- equal-weight mean of six DB drills;
- mean weighted by agreement with the 45° drill;
- a ridge model on all per-drill features fitted inside folds, in the style of Stuff+.

The single drill was best on held-out classes for all four outcomes (zone-matching closing +0.9%, CPOE +2.0%,
separation gained +0.3%, separation at throw +0.3%). Post-turn drive in other drills correlates only .06–.26 with it
(`results/75b`).

## Hypotheses for 2026 (directional, one-sided; player-permutation p unless stated)
- **D1 (primary, zone-match closing).** In 2026 zone-matching CB reps, Drive+ has a positive slope on closing.
  - Model: closing ~ Drive+ + receiver separation skill (other reps) + route.
  - 2023–25: +0.168 per 10 points, permutation p .017.
- **D2 (zone, not man).** The slope on closing is larger in zone-matching than in man-matching reps.
  - Interaction p < .05.
  - 2023–25: +0.147, p .04; man-matching slope ≈ 0.
- **D3 (throw side, team level).** On 2026 zone-coverage pass attempts with a graded CB lined up at CB or SLOT_CB on
  a clear side (|y| ≥ 4 yd from the field centre at the snap):
  - the Drive+ slope on completion for throws toward his side, minus the slope for throws away, is < 0;
  - model: + defense and offense team-season fixed effects;
  - 2023–25: −2.05 pp per 10 points, p .026. The away slope is the placebo (2023–25: +0.2 pp);
  - EPA is reported but not claimed (2023–25: 0.000).
- **D4 (time course).** In zone-matching reps, the fastest-minus-slowest Drive+ third separation difference is more
  negative at +1.0 s after the break than at +0.3 s.
  - 2023–25: −0.69 vs about −0.19 yd.
- **D5 (lasting).** For returning CBs (2023–25 classes) in the 2026 season, the D1 slope is > 0. Experience is not
  expected to erase it (2023–25: slope did not shrink with experience).
- **D6 (secondary, reported regardless).** Speed 1.0 s after the turn predicts zone-matching closing at least as well
  as Drive+.
- **Descriptive (reported regardless).** Completion % allowed and EPA per target by Drive+ tier.
  - 2023–25: 110+ tier 34%, <90 tier 76%.
  - The tier ends had only 6–8 CBs, so this is not a test.

## Registered as expected to stay null
- Multi-drill Drive+ (equal or learned) does not beat single-drill Drive+.
- Team-level scheme fit: drive × team Cover 3/4/6 share (2023–25 p .61–.74) and drive × the team's tracking
  zone-match rate, leaving the CB out (2023–25 p .23–.86).
- Head-around: facing the ball at release or arrival.
- The 90° break-to-ball drill's drive phases.
- Throw-side EPA (as opposed to completion).

## Samples and decision
- Minimums:
  - D1/D2: ≥ 25 CBs and ≥ 300 zone-matching reps.
  - D3: ≥ 30 CBs and ≥ 5,000 throw rows.
  - Below these, the result is "inconclusive".
- 2026 rookies are graded with `drive_plus.py`. Returning players use the frozen CSV.
- **Decision:**
  - **Confirmed:** D1 and D3 both pass; they are two independent units, rep level and throw side.
  - **Partial:** one passes.
  - **Retired:** neither passes and both point estimates have the wrong sign.
