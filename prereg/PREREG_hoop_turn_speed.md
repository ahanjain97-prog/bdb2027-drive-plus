# Pre-registration: Hoop Turn Speed → NFL pass rush (frozen 2026-10-12)

This file was written before any 2026-season game data or 2026-class combine data was seen. Nothing below may be
changed after that data arrives. If it ever is changed, the change must be dated and described here.

## Frozen artifacts
| file | SHA-256 |
|---|---|
| `hoop_turn_speed.py` (metric definition) | `903c65f7fca25d4c038eee3d3f56ec7ef943f675f421761decceb68f03b1b5eb` |
| `hoop_turn_speed_frozen_2026-10-12.csv` (values for the 109 DL with hoop data, 2023–25 classes) | `b7805267df748547ff56fb84a1d441d14cbf89b5bb4ecd6a9111e43c4d41bac0` |

The metric is the mean speed (yd/s) a defensive lineman carries through the turning sections of Run the Hoop. A
turning section is one where lateral acceleration exceeds 1.5 yd/s² at more than 1.5 yd/s. Sections are
duration-weighted, then the player's attempts are averaged. The analysis code in `code/` reproduces this file exactly.

## Evidence that motivated it (2023–25 seasons; these data cannot be reused as the test)
- **Hoop turn speed → pass-rush penetration over expected** (69 DL, ≥ 50 REG+POST rush snaps):
  - r +.52 raw; +.38 after body weight; +.26 after weight plus all six stopwatch tests.
  - By class: 2023 +.38, 2024 +.55, 2025 +.47. Replicates in both halves of games.
- **Head-to-head reps** (6.6K engaged DL–OL reps): the rusher-bend × blocker-quickness interaction is negative on
  rusher pressure, blocker peak pressure probability and blocker pressure allowed (player-permutation p .012–.030).
- **Not found:** a main effect on recorded pressure (r +.09).

## Hypotheses for the 2026 test (directional, one-sided)
- **H1 (primary).** Among DL with hoop data and ≥ 50 REG+POST pass-rush snaps in 2026, hoop turn speed correlates
  positively with 2026 pass-rush penetration over expected. Target: r > 0, one-sided p < .05.
- **H2.** H1 holds after controlling for alignment unit (EDGE/IDL) and combine weight (partial r > 0, one-sided
  p < .05).
- **H3.** In 2026 engaged DL–OL reps, the rusher hoop × blocker five-yard-wave-speed interaction on blocker peak
  pressure probability allowed is negative. Same model as `code/31_matchup_model.py`. One-sided p < .05 by the
  player-permutation test of `code/32_interaction_check.py`.
- **H4 (exploratory, reported regardless).** Hoop turn speed → 2026 pressure over expected, r > 0.

## Outcome definitions (fixed)
Exactly as in `code/21_bend_game.py`, `code/23_bend_deep.py`, `code/30_matchups.py` and `code/31_matchup_model.py`
as of the date above:
- Penetration = deepest point behind the LOS before the throw or the play ending, within 3.0 s.
- "Over expected" = minus the leave-player-out mean of its cell (alignment × blitz × down × distance bucket).
- Pressure = time_to_pressure recorded or a sack.

## Samples
- **2026 rookies (2026 class):** compute the metric with the frozen file and test H1 and H2 within the class.
- **Returning players (2023–25 classes):** use the frozen values above and their 2026 outcomes.

Report both samples, plus the pooled result as primary. Minimum for a valid test: 30 DL in the pooled sample.
Below that, report as "inconclusive".

## Decision
Claim "hoop turn speed predicts NFL pass-rush penetration" only if H1 and H2 both pass. If H1 passes and H2 fails,
claim "hoop turn speed tracks NFL rush depth, largely through body mass". If H1 fails, the metric is retired.
