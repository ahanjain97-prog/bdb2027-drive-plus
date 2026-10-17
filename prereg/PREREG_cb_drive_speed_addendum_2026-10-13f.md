# Addendum f to PREREG_cb_drive_speed.md: framing, criterion, and the after-throw channel (written 2026-10-13)

Written before any 2026-season game data or 2026-class combine data was seen. The frozen metric, the primaries D1 and
D3 (addendum d) and the wide sample and evidence tiers (addendum e) are unchanged. This addendum fixes three things
about how results will be judged and reported, so they cannot be chosen after the fact.

## 1. The claim is a zone-match claim
Drive+ is a grade for corners **matching a receiver in zone coverage on routes where the receiver keeps running after
his key move** (ins, posts, corners, verticals, crossers). In 2023–25 the per-rep effect on the composite coverage score
was +0.09 SD per 10 points in zone-matching reps and +0.02 (not supported) in man-matching reps. The 2026 report will
state zone-matching and man-matching results side by side; a null in man is **expected**, not a failure. A route split
(continuing routes vs stop routes: HITCH, SLANT) is reported; the 2023–25 pattern (edge on continuing routes, none or
reversed on hitches) is exploratory and carries no decision weight.

## 2. The primary criterion for the movement signal is the composite coverage score
Per rep, the mean of the available context-adjusted z-scores of: separation gained at the key move (−), closing after
it (+), separation at the throw (−), receiver open at the throw (−); each adjusted for receiver skill, zone/man, route
and position, exactly as in `code/90_signal_ceiling.py`. In 2023–25 its player-mean reliability was .70 (single
outcomes .57–.68). The 2026 movement result is reported as the Drive+ slope on the composite, with an 80% interval,
plus the observed and disattenuated player-level correlations (2023–25: r .29 observed, ≈ .42 corrected for Drive+
reliability .66 and criterion reliability .70). Significance tallies across correlated outcomes are not used as evidence.

## 3. The after-throw channel is a separate hypothesis
In 2023–25, two-thirds of the completion effect on targets (−9.8 of −15.1 points, 110 vs 90) occurred after the ball
was released, with no measured mechanism (breakups and head-around were null) and a concentration in the 2025 season.
It is therefore registered on its own:
- **A1.** On 2026 targets of his man, completion over the expected completion (the cross-fitted xComp model of
  `code/79_beyond_completions.py`, refit on 2026 pool-receiver targets) declines with Drive+ (one-sided).
- **A2.** On 2026 zone throws to his side (76b rows), the targeted receiver is open (NGS separation ≥ 2 yd) less often
  than on throws to the other side, per 10 Drive+ points (2023–25: −2.9 points, permutation p .036).
The movement claims (D1, W1–W4) are judged **independently** of A1 and A2. A failure of A1/A2 does not count against
them; a success of A1/A2 does not rescue them.

## 4. Reporting
Every estimate is reported as a point value with an 80% interval and the evidence tier of addendum e; 2024-type and
2025-type seasons are reported separately; the three seasons' pooled estimate is the headline only when the seasons
agree in sign.
