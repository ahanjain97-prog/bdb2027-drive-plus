# NFL Big Data Bowl 2027: Drive+ and combine-to-NFL movement research

Two lines of work on the 2027 Big Data Bowl data (10 Hz combine tracking for the 2023–25 classes, 2023–25 game
tracking for the 510-player pool):

1. **Drive+** (the submission). A frozen, rule-based combine grade for defensive backs: top speed in the drive after the
   turn in the backpedal-and-transition (45°) drill, z-scored within the combine class and put on a 100 scale.
   On Sundays it is a zone-match trait.
2. **Self-supervised movement signatures** (the original project, below): a masked-trajectory transformer and probe
   harness. Fingerprints yes, outcomes no.

## Drive+ in one paragraph

On 2,952 NFL coverage reps where a graded DB was on a tracked receiver before the receiver's break (76 DBs, 502 targets),
each 10 Drive+ points means 0.16 yd less separation gained at the break, 4.6 points less often open at the throw and
8.5 points lower completion % when targeted (110 vs 90: −17 points; −15 with receiver separation and catch skill held
fixed). The coverage effects replicate in 2024 and 2025. About a third of the catch effect runs through separation at
the throw; the rest happens after the ball is in the air. Bottom-third corners give up extra catches beyond their
separation in both seasons; top-third corners take catches away beyond their separation in 2025 only, which is small
enough to be chance. In man coverage the grade predicts little. Everything is pre-registered for the 2026 season.

| | where |
|---|---|
| Kaggle writeup (markdown, figures, notebook, PDF) | `submission/` |
| Interactive page (published) | template `code/97_nfl_page.html`, data `code/97_nfl_page_data.py`, build `code/97_build_nfl_page.py <out.html>` |
| Pre-registration (metric, frozen grades, hypotheses, addenda c–f) | `prereg/` |
| Every script, one line each | [`docs/SCRIPTS.md`](docs/SCRIPTS.md) |
| Results (text / csv / figures / page data) | `results/` |

### The Drive+ chain

| step | scripts |
|---|---|
| the metric (frozen) | `prereg/cb_drive_speed.py`, `prereg/drive_plus.py` → `prereg/drive_plus_frozen_2026-10-12.csv` |
| coverage reps, closing, zone-match cell | `41_coverage_reps.py`, `57_cb_mechanism.py`, `70_cb_match_zone.py` |
| which moment of the drive matters | `72a`–`72d` (decomposition, components, controls, ball flight) |
| separation clock, throws toward vs away | `73g_separation_curve.py`, `76b_toward_away.py`, `77_*` |
| every-rep outcomes, xComp, hardening | `78`–`83` |
| audit, registered wide sample, ceiling | `88_audit.py`, `89_wide_validation.py`, `90_signal_ceiling.py`, `91_within_data.py` |
| page and submission figures | `92_viz_wide.py`, `94_submission_figs.py`, `95_build_notebook.py`, `96_football_figs.py`, `97_*` |
| outside-data check (nflverse, NGS) | `93_nflverse_ext.py` |
| CPOE, EPA per target, season split, receiver skill | `98_cpoe.py`, `98b_cpoe_mediation.py`, `99_epa_target.py`, `100_season_why.py`, `101_sep_to_catch.py`, `102_after_throw_seasons.py`, `103_wr_skill_controls.py` |

### Reproducing
- Put the competition files in `data/raw/` as parquet (`code/00a_to_parquet.py` converts the Kaggle CSVs), then build
  the sequence tensors with `code/01_build_sequences.py`. `data/` is not in the repo: competition data can't be
  redistributed.
- Scripts use absolute paths rooted at `/Users/ahanjain/bdb_ssl` (and `/Users/ahanjain/nfl_data/raw` for the optional
  nflverse check). Editing those paths is fine for a run, but don't commit edits to the pinned files below.
- `python tools/verify_prereg.py` recomputes the SHA-256 of every file pinned in the pre-registration (14 files) and
  fails if any changed.
- Python packages: `requirements.txt`.

---

# Self-supervised movement signatures (original project)

A masked-trajectory transformer is pretrained on every tracked player-play (311,716 sequences, 10 Hz,
snap −1.0 s … +6.9 s). Each play's embedding is turned into a context-adjusted player **signature**, and the
signature is tested against outcomes it never saw. Every result is compared against a hand-built physics
block (16 features) and a combine MiniRocket block.

Run everything with `./run_all.sh` (≈2 h on an M5; pretraining dominates). The scripts are in `code/` and are
numbered in pipeline order.

## Pipeline

| step | script | what |
|---|---|---|
| 0 | `00a_to_parquet.py`, `00_inspect.py` | typed parquet; conventions verified (dir/o: 0° = +y, clockwise → vx = s·sin(dir); `a` is unsigned) |
| 1 | `01_build_sequences.py` | rotate `left` plays 180°, x relative to LOS, y relative to snap spot; signed along-track + lateral accel derived; 80-frame tensors |
| 2 | `02_pretrain.py`, `model.py` | 6-layer d192 transformer (2.7 M params) + one context token (alignment, route, coverage, formation, down/dist…; **no outcomes**). Span masking + 25% forecast masking, 15% orientation dropout, 15% context dropout |
| 3 | `03_embed.py` | per-play embedding = mean frame token, **averaged with the left/right-mirrored play** (see traps); "surprise" = masked-reconstruction loss |
| 4 | `04_physics.py` | 16 hand-built per-play features (top speed, max accel/decel, max lateral accel, backpedal/sideways share, depth, …) |
| 5 | `05_targets.py` | targets: REC sep / targets per route / yards per route; OL peak pressure prob / pressure allowed; DL pressure rate / get-off; DB tackle rate; snaps per game; career snaps, starts, honors |
| 6 | `06_signatures.py`, `sig_utils.py` | player × {game half, season, rookie year} signatures; each play minus the **leave-player-out** mean of its context cell (side·alignment·play type·route/assignment) |
| 7 | `07_combine_minirocket.py` | MiniRocket (10K kernels) on speed / accel / along-track / lateral channels per drill, label-free PCA |
| 8 | `08_probe.py`, `08b_probe_selftest.py` | ridge probes with permutation nulls (below) |
| 9–12 | `09_reliability.py`, `10_summary.py`, `11_interpret.py`, `12_ol_passpro_check.py` | stability / fingerprint / novelty; BH summary; dimension descriptions; robustness of the best hit |

### Probe design
- **half**: signature from a random half of games → target on the other half. `half_rev` swaps the halves to replicate.
- **forward**: signature from season t → target in season t+1. **career**: rookie-season signature → career outcomes.
- **Baselines**:
  - `draft`: subgroup + pick/UDFA + age + conference + measurables + combine tests + NGS scores + log(#plays).
  - `prior`: `draft` plus the same target measured on the signature data.
  - `prior+phys`: `prior` plus the 16 physics features.
- **Model**: block → label-free PCA(8). Ridge on [baseline, c·PCs], with alpha and the block scale c chosen by exact leave-one-out error. c = 0 nests the baseline. Repeated 5-fold CV, one row per player.
- **Null**: permute block rows within subgroup and refit everything.
- **Self-test** (`08b`): a planted signal is detected in all three positive controls (p = .03, .01, .01). Noise blocks give uniform p-values.

## Results (2026-10-11)

**Movement model.** Masked-frame validation error is 0.092 vs 0.223 for linear interpolation (−59%). It is
best on position and velocity and worst on orientation rate.

**Signatures are stable and individual** (`results/09_reliability.txt`, split-half, ~90–98 players per group):

| block | top-8 PC reliability | fingerprint top-1 (chance ≈ 1%) | variance not explained by physics + measurables |
|---|---|---|---|
| SSL (mirror-invariant, context-adj.) | .54–.73 | 36–53% | 25–32% |
| physics (context-adj.) | .59–.77 | 45–55% | — |

After the mirror fix, the large learned dimensions are re-expressions of simple physics. Examples: REC PC1 =
top speed (r .91), REC PC2 = backpedal share (.87), DL PC1 = peak acceleration magnitude (.95), DB PC2 =
alignment depth (.84). Only REC and OL keep one dimension each that is both reliable (r > .5) and mostly
unexplained.

**Outcomes — no robust learned signal** (`results/10_summary.txt`, 516 tests):
- **Primary family** (SSL, half + forward, 60 tests): 6 tests at BH q = .05. All are held-out-game OL pass
  protection or DL get-off, i.e. movement describing a movement-derived stat in the same season.
- **The best hit** (OL pressure allowed, beyond draft + track record + physics: +0.054) **does not survive**:
  - With 20 CV repeats the gain is −0.016.
  - With the halves reversed it is ≤ 0 on every baseline (`results/12_ol_passpro_check.txt`).
- **Next season** (forward): SSL adds nothing beyond draft or beyond track record in any group.
- **Career**: no block adds to draft information (snaps, starts, honors).
- Raw (not context-adjusted) blocks score more p < .05 hits because they carry role and alignment.

**Leads only.** Both are forward design, small n, with no internal replication; test on the 2026 season:
- **Surprise** (how hard the model finds a receiver's movement to predict) → next-season snaps per game, REC:
  +0.074 over draft, +0.047 over draft + current snaps (p .005 / .010, n 61). Wrong sign for DL/DB.
- **Combine MiniRocket** → next-season targets per route, REC: +0.086 / +0.091 (p .005, n 54). Yards per
  route turns negative once the prior is added.

## Traps hit (and fixed)
1. **Left/right side leak.** The first embedding's largest PCs were formation side: left vs right tackles
   (OL PC2, 34% of variance), and the same for DL and DB. The physics features are sign-invariant, so these PCs
   looked "novel". Fix: average each play's embedding with its y-mirrored copy, flipping y_rel, y_field, vy,
   a_lat, cos_o, sin_od and o_rate, and swapping LEFT/RIGHT coverage assignments. The old version is kept as
   block `sslasym`.
2. **Stacked rows leak through LOO alpha selection.** Two rows per player have near-identical signatures, so
   leave-one-out picks tiny alphas and memorizes the player (ΔR² down to −0.95). Fix: exactly one row per player.
3. **A shared ridge penalty kills power.** A noise block cost about −0.1 R², which masked the planted
   positive control. Fix: a separate block scale c in {0, .25, .5, 1, 2}.
4. **p < .05 is not the same as gain.** Random blocks hurt R², so the null mean is below 0. Only 3 of the
   58 p < .05 tests had ΔR² ≤ 0, but always read p together with ΔR² and the replication direction.
5. **MPS:** keep the data tensors on the GPU. CPU-side indexing ran at 0.3 it/s vs 2.4. MPS memory is capped
   at half of RAM via the watermark env vars.

## Next steps
- **Re-test the two leads on the 2026 season** before believing either.
- **Player-conditioned pretraining.** Add a per-player token so the "beyond context" signal is learned
  directly instead of averaged after the fact.
- **Encoder on combine drills:** done (`14_combine_ssl_embed.py`). It matches hand-built physics and
  MiniRocket for combine→game translation but does not beat them (see the translation section).

## Combine → NFL game tracking translation (2026-10-11, scripts 13–17)

The question is how a player's combine movement shows up in his NFL game movement. Draft information and track
record are left out. Body size and position are the only reference, because both sides reflect body mass. Players
in the sample have combine tracking and at least 80 tracked NFL plays (about 105 per group).

**1. Direct comparison: same quantity, same units, same formulas** (`13_translation_direct.py`, per-player table
in `results/13_translation_per_player.csv`). Game capacity = p99/p95 of per-play maxima. Combine capacity = max
over attempts. Reliabilities: combine = alternate attempts; game = halves of games.

| within position | WR | TE | T | IOL | EDGE | IDL | CB | S |
|---|---|---|---|---|---|---|---|---|
| top speed r (combine 40 → game) | .64 | .56 | .35 | .17 | .43 | .28 | .45 | .19 |
| game top speed as % of combine | 86% | 81% | 63% | 64% | 74% | 72% | 84% | 83% |
| lateral accel r | .13 | .37 | .29 | −.06 | .17 | .19 | .15 | .23 |
| peak accel r (40 start) | −.08 | −.05 | .06 | −.09 | −.04 | .34 | .01 | −.21 |

- **Top speed is the one quantity that transfers.** The combine measurement is near-perfectly repeatable (.94–.97).
- **Single-frame peaks do not transfer, mostly because the combine does not measure them reliably.** Peak decel
  has reliability of about 0 for WR, TE, IOL and IDL. Peak accel at the 40 start is negative attempt-to-attempt
  for WR.

**2. One combine feature at a time, beyond position + size** (`16_single_feature_check.py`).
- Tracked 40 top speed adds R² +.053 to REC game top speed and +.112 to DB.
- The official 40 time adds less: +.024 and +.095.
- No single accel, decel or lateral number adds anything in any group.

**3. Whole movement profile** (`15_translation_model.py`). The profile has 29 targets: 5 capacities, 16
context-adjusted style features and 8 learned game-signature PCs. The sources are stopwatch, drill physics,
MiniRocket, and the game-pretrained transformer applied to the combine drills.
- **In-game peak acceleration is predictable from combine tracking.** Beyond size, MiniRocket adds R² +.17
  (REC), physics +.11 (OL) and physics +.11 (DB); the transformer adds +.11 (REC). The stopwatch adds nothing.
- **Nothing else translates.** Style features and learned game signatures get ΔR² ≈ 0. DL capacities are
  already .55–.71 R² from size alone, and combine data adds nothing to that.
- **Lineup test** (pick the player's NFL profile from his position group):
  - Combine data ≈ size alone: REC 70th vs 69th percentile, DL 64th vs 64th, OL and DB at chance.
  - Top-5 hit rate for REC rises from 12% to 16% with the transformer.
- **Big blocks dilute single strong features.** Top speed is lost inside multi-drill blocks. For matched
  quantities, use the direct one-to-one feature.

**4. Which drills carry in-game acceleration** (`17_accel_drills.py`; partial r beyond position + size, BH within
group). Sustained speed through position drills transfers. Single-frame peak accel in drills does not, or carries
a negative sign.
- **DB:** backpedal-and-45°-transition mean speed .38 (q .005).
- **OL:** five-yard wave (slide & shuffle) mean speed .35 and speed at 1 s .33 (q .016); screen drill peak accel .32.
- **WR:** slant route mean speed .39 (q .04); curl and comeback peak decel −.34 and −.32.
- **DL:** nothing survives.

## Bend: pass rushers, OL, LBs (2026-10-12, scripts 20–27)

Bend is defined physically. Lateral (centripetal) acceleration while turning at speed; lean = atan(a_lat / g).
Combine and game use the same formula and the same 3-frame smoothing. Combine drills: Run the Hoop (a figure-8,
so both turn directions), Pass Rush Drill (one rep per side), and OL pulls (both directions).

**What predicts (DL, edge + interior pooled, n 69–91):**
- **"Hoop turn speed"** (speed carried through the hoop's turning sections) → NFL pass-rush penetration over
  expected:
  - r +.52 raw.
  - +.38 after body weight (play level +0.075 yd/SD, p .07).
  - +.26 after weight + 40 / 10-yd / 3-cone / shuttle / vertical / broad (p .03).
  - By class: 2023 +.38, 2024 +.55, 2025 +.47. Leave-one-out +.50 to +.55. Replicates in both halves of games.
  - Penetration (max depth before the throw) predicts pressure on the same play (AUC .785) and is twice as
    reliable as pressure (.56 vs .29).
- **Hoop time** → NFL arc speed around the corner: r −.54 raw (−.32 after unit).
- **IDL only:** pass-rush-drill sustained lean → NFL corner bend, r +.43 (replicates).

**What doesn't:**
- Peak lean (the classic single-frame "bend" number).
- Hoop turn speed → recorded pressure: r +.09; play-level OR 1.06 per SD (p .37). Pressure's reliability is .29.
- Bend → run stopping (tackles, TFL, run penetration): ≈0.
- Side-matched asymmetry: combine left−right bend does not predict NFL left−right bend.

**OL** (216 tests, none survives BH):
- Five-yard wave drill speed → NFL pull bend, r +.39 (replicates, weight-robust).
- Pull-drill bend → climbs, pulls, run yards and pass pro: ≈0.

**LBs:** untestable. The pool has 2 ILB + 1 MLB. Off-ball snaps belong to box safeties and nickels. The DL
block-shed proxy (stall at the LOS, then speed regained) is only weakly validated (→ run tackles r +.23, p .14).

**Unique asset found:** 13,296 plays where a pool OL and a pool DL align within 1.5 yd at the snap (7,316
dropbacks; 117 OL, 107 DL). These are true one-on-one reps with both trajectories tracked.

## Head-to-head reps, where bend matters, pre-registration, box safeties, DE importance, LT/RT (2026-10-12, scripts 30–36)

**Head-to-head** (`30_matchups.py`, `31`, `32`). 9,322 engaged DL–OL rush reps (both players tracked, min
distance ≤ 1.5 yd), 113 DL / 116 OL. "Corner turned" (the rusher gets deeper than the blocker while at his hip)
changes outcomes:
- Blocker's peak pressure probability: .26 → .42.
- Rusher pressure: 6% → 21%.

**Rusher bend × blocker quickness** (hoop turn speed × five-yard-wave speed) is negative:
- rusher pressure −1.05 pp per SD², player-permutation p .020;
- blocker pressure probability −.0096, p .012;
- pressure allowed p .030.

Rusher pressure rate by bend tercile and blocker quickness:

| | slow-footed blocker | quick-footed blocker |
|---|---|---|
| high-bend rusher | 12.4% | 8.0% |
| low-bend rusher | 7.3% | 8.3% |

It is strongest at G/C vs IDL. It is robust to weight, arm-length and 10-yd controls, and significant in 2025 alone.

**Where bend matters** (`33`). Hoop → penetration nearly doubles when the QB holds the ball ≥ 2.8 s: +0.19 vs
+0.10 yd/SD, interaction p .014. There is no amplification on obvious passing downs, late in drives, under
fatigue or in the 4th quarter.

**Pre-registration** (`prereg/`). A frozen metric file, the frozen 109-DL leaderboard (SHA-256 recorded), and
four directional 2026 hypotheses with decision rules.

**Box safeties** (`34`, n 40). In-game downhill speed after the handoff → TFL r .52 and fewer yards on his
tackles r −.51. Combine DB turn speed points the right way (r ≈ .19) but is n.s. Underpowered.

**Edge-rusher importance** (`35`, n 39–42).
- Combine → penetration: broad jump .51, hoop turn speed .41 (best after weight: .37), 40 .45, 10-yd .38,
  peak lean .27.
- NFL skills in one half of games → penetration in the other: get-off .52 > arc speed .39 > apex speed .37 >
  lean .33.
- Nothing predicts recorded pressure.

**LT / RT** (`36`). Side comes from field geometry: the tackle's position relative to the field center at the
snap. Per-season consistency is .99, and it recovers real switches (Paris Johnson RT→LT 2024, Broderick Jones
RT→LT 2025, Joe Alt RT→LT 2025).
- No LT/RT difference in pressure probability allowed (context-adjusted −.012, p .23; within 11 tackles who
  played both sides +.003, p .82).
- LTs and RTs face rushers with the same bend.
- Rusher bend is not more damaging on either side.
- Trait × bend interactions by side are noise-level (3 of 20 at p < .05, mixed signs).
- LTs: taller, longer tackles allow less pressure (height r −.69, n 12). Fragile.
