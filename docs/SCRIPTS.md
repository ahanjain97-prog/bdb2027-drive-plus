# Script index

Generated from each script's docstring (`code/`, numbered in the order the work was done).

| script | what it does |
|---|---|
| `00_inspect.py` | Raw-data checks before building sequences: frame rate, events, dir/o convention, joins. |
| `00a_to_parquet.py` | One-time CSV -> parquet conversion (typed, zstd) so later steps don't re-parse 2 GB of CSV. |
| `01_build_sequences.py` | Build fixed-length, play-standardized trajectory tensors for every tracked player-play. |
| `02_pretrain.py` | Self-supervised pretraining: masked-span + forecast reconstruction of player trajectories. |
| `03_embed.py` | Per-play outputs of the pretrained encoder. |
| `04_physics.py` | Hand-built per-play physics features: the fair baseline the learned embedding has to beat. |
| `05_targets.py` | Outcome targets per player x data part, plus draft-time baseline covariates. |
| `06_signatures.py` | Build player x part signatures for a per-play block (phys / ssl / surprise). |
| `07_combine_minirocket.py` | Combine-drill block: MiniRocket random-convolution features per drill attempt -> player x drill. |
| `08_probe.py` | Probe harness: does a feature block predict NFL outcomes beyond what was known at the draft? |
| `08b_probe_selftest.py` | Harness self-test: a block carrying real signal must be detected; pure-noise blocks must be null (p ~ U(0,1)). |
| `09_reliability.py` | Are the signatures stable, individual, and novel? (No outcomes used.) |
| `100_season_why.py` | Why do the catch outcomes (completion, EPA per target) show only in 2025 when the movement outcomes show in both seasons? |
| `101_sep_to_catch.py` | How much separation before the throw do high-Drive+ DBs take away, and how many catches is that separation worth? |
| `102_after_throw_seasons.py` | How can the after-throw catch effect be ~0 in 2024 and large in 2025? |
| `103_wr_skill_controls.py` | (103) Control for receiver skill: separation skill (wr_skill) and catch skill (leave-one-out catches over expected). |
| `10_summary.py` | Summarize probe results: Benjamini-Hochberg over the pre-declared primary family, then secondary tables. |
| `11_interpret.py` | Describe the learned signature dimensions in football terms. |
| `12_ol_passpro_check.py` | Robustness of the one primary hit: OL pressure_allowed, ssl:resid beyond draft + prior + physics (half design). |
| `13_translation_direct.py` | Direct combine -> NFL comparison of the SAME physical quantity, same units, same formulas. |
| `14_combine_ssl_embed.py` | Embed every combine drill attempt with the GAME-pretrained encoder (shared representation space). |
| `15_translation_model.py` | How much of a player's NFL game movement can his combine data predict? |
| `16_single_feature_check.py` | Do multi-feature blocks dilute a single strong combine feature? Same CV/ridge machinery as 15, one feature at a time. |
| `17_accel_drills.py` | Which combine drill movements carry in-game acceleration? Partial r (given subgroup + height + weight + arm) |
| `20_bend_combine.py` | Combine BEND, by turn direction, from the drills that contain real turns at speed. |
| `21_bend_game.py` | In-game bend and related mechanics, per play, plus the play's outcome. Same lateral-accel formula and |
| `22_bend_tests.py` | Bend hypotheses: combine bend -> NFL movement and NFL performance for edge rushers, interior DL and OL. |
| `23_bend_deep.py` | Bend, second pass: sustained metrics, penetration outcome, full combine x game matrix for DL, replication filter. |
| `24_hoop_robustness.py` | Robustness of the hoop -> pass-rush penetration link (DL pooled, unit-adjusted). |
| `25_ol_bend.py` | OL: combine bend / redirect -> NFL pulls, climb blocks, pass protection. Same design as 23 (matrix + both-halves |
| `26_pressure_glm_matchups.py` | (1) Play-level pressure model: pressure ~ hoop turn speed (player covariate) + unit + context cells, logistic, |
| `27_checks.py` | consistency checks |
| `30_matchups.py` | Head-to-head pass-rush reps: a pool DL (DE/OLB/DT/NT aligned EDGE or INTERIOR_LINE) vs a pool OL (T/G/C) on the |
| `31_matchup_model.py` | Does the rusher's combine bend vs the blocker's combine quickness predict who wins the rep? |
| `32_interaction_check.py` | Robustness of rusher-bend x blocker-quickness interaction: player-level permutation null (shuffle each player's |
| `33_where_bend_matters.py` | Where does bend matter most? Play-level, all DL rush snaps (REG + POST): outcome ~ hoop turn speed x condition. |
| `34_box_safeties.py` | Box defenders as the linebacker stand-in: DBs (SS/FS/CB/DB) and the few ILB/MLB aligned off the ball in the box |
| `35_de_bend_importance.py` | How important is bend for edge rushers (DE/OLB aligned EDGE)? |
| `36_lt_rt.py` | LT vs RT in pass protection. |
| `40_outcome_library.py` | Phase 1a: player-level NFL outcome library by position, context-adjusted, with split-half reliability. |
| `41_coverage_reps.py` | Phase 1b: coverage reps -- a pool WR/TE running a route and a pool DB covering him on the same dropback, both tracked. |
| `42_tackle_reps.py` | Phase 1c: open-field tackle reps -- a pool receiver/back catches the ball and a pool defender records the tackle or |
| `50_feature_factory.py` | Phase 2: combine feature factory -- position- and drill-specific mechanics per attempt, then stability gating. |
| `50_feature_factory_v0.py` | Phase 2: combine feature factory -- position- and drill-specific mechanics per attempt, then stability gating. |
| `51_stability_experiment.py` | Can better estimation make combine mechanics repeatable? Test-retest (attempt 1 vs 2, drills with >= 15 repeaters) |
| `52_feature_catalog.py` | Phase 2b: per-position feature catalog (reliable features only) + per-position outcome table. |
| `53_tierA.py` | Phase 3, tier A: pre-registered matched hypotheses (drill mechanic -> the same mechanic's NFL payoff). |
| `54_tierBC.py` | Phase 3, tiers B and C. |
| `55_leads.py` | Stress-test the leads from tier A / B (all post-hoc relative to the screen -> leads, not confirmed findings). |
| `56_cb_transition.py` | Deep dive: CB backpedal-and-transition (45-degree reaction) drill -> separation at the receiver's break. |
| `57_cb_mechanism.py` | CB Transition Drive Speed: in-game mechanism, moderators, and quantified outcomes. |
| `58_cb_backtest.py` | CB Transition Drive Speed: ball-flight closing, out-of-sample backtests, robustness, and a completion model. |
| `59_cb_more.py` | More checks and hypotheses around CB Transition Drive Speed. |
| `60_game_transition.py` | Step 1: the in-game version of the transition drill, measured on EVERY CB dropback snap (no receiver needed). |
| `61_game_transition_adjusted.py` | Step 1b: context-adjusted in-game transition drive, capacity percentiles, and the bridge to coverage outcomes. |
| `62_cb_value.py` | Step 2: the full value of CB Transition Drive Speed -- target avoidance, completion over expected allowed, EPA per rep. |
| `63_sprint_profile.py` | Step 3: sprint physics of the drive phase vs the same player's 40. |
| `64_profile_hier.py` | Steps 4 + 6: two-dimensional coverage profile (zone closing / man mirroring) and a crossed random-effects model. |
| `64b_cb_variance.py` | Step 6 (robust version): how much of the true between-CB difference in coverage does drive speed explain? |
| `65_projection.py` | Step 5: projection backtest -- what is combine drive speed worth when a team has only k NFL coverage reps on a CB? |
| `66_cb_full_features.py` | Full, unshrunk CB combine feature table (no reliability filter, no selection, no PCA). |
| `66b_cb_full_features_v2.py` | Full CB combine features, v2 -- comparable across combine years. |
| `67_cb_full_models.py` | Full unshrunk CB combine data -> NFL coverage. Which model uses the combine best, out of sample? |
| `67b_cb_full_models_v2.py` | v2 (year-comparable features, draft class in the context): Full unshrunk CB combine data -> NFL coverage. Which model uses the combine best, out of sample? |
| `68_cb_drill_breakdown.py` | Where in the full CB combine data is the coverage information? |
| `68b_cpoe_drills_v2.py` | Which drills carry the completion-over-expected information in the year-comparable (v2) features? |
| `68c_turn_curve_readout.py` | Which moment around the transition-drill turn predicts completions allowed? Ridge (grouped-CV alpha) on the v2 |
| `69_drive_contrast.py` | The shape the full model found, as one number: how much faster a CB comes OUT of the transition than he went INTO it. |
| `70_cb_match_zone.py` | Is CB Transition Drive Speed a match-zone trait? |
| `71_cb_sink_bend.py` | Sink, bend, hip flip and drive for CBs: combine -> NFL movement -> coverage outcomes, zone-matching vs man. |
| `71b_sink_bend_followup.py` | Follow-ups to 71 (CB sink / bend / hip flip). |
| `72a_drive_decomposition.py` | Where inside the transition drill does CB Transition Drive Speed live? |
| `72b_drive_parts_outcomes.py` | Which phase of the drill drive carries the NFL effect? + time-matched translation. |
| `72c_drive_controls_predict.py` | CB drive speed: (1) mechanistic prediction, (2) controls ladder, (3) out-of-sample prediction. |
| `72d_ball_flight.py` | Is the drive effect a break on the THROWN BALL? (72c: drive's completion effect is largest on throws released < 0.5 s |
| `73a_head_around.py` | Step 2 -- head-around: are high-drive CBs turned to the ball sooner? (the completion effect 72d could not explain) |
| `73b_bp90_phase.py` | Step 3 -- second drill, same phase: BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL. |
| `73c_lb_zone.py` | Step 4 -- linebackers: does the same late-drive phase translate for zone droppers? |
| `73d_experience.py` | Step 5 -- does experience close the gap? Drive-speed slope by NFL experience (season - draft class: 0 rookie, 1, 2). |
| `73e2_scheme_fit_tracking.py` | Step 6b -- scheme fit with a TRACKING-defined team measure instead of shell labels. |
| `73e_scheme_fit.py` | Step 6 -- scheme fit: is drive speed worth more to a CB whose TEAM plays more match-type zone? |
| `73f_wr_phase.py` | Step 7 -- the receiver side of the same phase, and the phase-matched matchup. |
| `73g_separation_curve.py` | Step 8 -- the headline chart: receiver-CB separation around the receiver's break, by combine drive-speed tercile. |
| `74_team_epa.py` | Team-level validation: when a defense plays ZONE, is it better (lower offensive EPA) with a high-drive CB on the field? |
| `75a_all_transitions.py` | Drive+ step 1: every transition in every DB drill -> post-turn drive speed. |
| `75b_drive_plus.py` | Drive+ : a Stuff+-style headline grade (100 = average CB, 10 = 1 SD), single-drill vs multi-drill vs learned. |
| `75c_drive_plus_card.py` | Drive+ grade card: grade every CB (100 = average, 10 = 1 SD) and check NFL outcomes by grade tier. |
| `76a_drive_plus_all_db.py` | Drive+ for every defensive back who ran the 45-degree transition drill (CBs + safeties + DBs), on the CB scale, and a |
| `76b_toward_away.py` | Throws TOWARD vs AWAY from the high-Drive+ corner: does the defense allow less EPA on throws to his side? |
| `77_epa_decomposition.py` | Why does throw-side EPA not move when throw-side completion does? (76b: zone completion toward a high-Drive+ CB's side |
| `77b_depth_check.py` | Follow-up to 77: offenses throw DEEPER to a high-Drive+ CB's side. Is the completion drop just the depth shift? |
| `78_grade_card_all_db.py` | Drive+ grade card on the biggest reasonable sample: all graded DBs (CB/SS/FS/DB), all alignments, 2023-25 REG+POST. |
| `79_beyond_completions.py` | Beyond completions: score EVERY coverage rep, check which metrics are stable, and backtest small-sample shrinkage. |
| `80_visuals.py` | Football-specific visuals for Drive+ (figures 1-5; 6-7 in 80b). Output results/figs/*.png |
| `80b_visuals.py` | Visuals 6-7 for Drive+ (registered 3-yd reps, all graded DBs; per-rep expected completion from 79). |
| `81a_metric_hardening.py` | Harden the Drive+ measurement: robustness of the grade to every construction choice, and which version is most reliable. |
| `81b_pbp_credits.py` | Biggest outcome sample: play-by-play coverage credits for every pass snap a graded pool DB played (any receiver). |
| `81c_spec_curve.py` | Is Drive+ fragile? Specification curve over every reasonable analysis choice + influence diagnostics. |
| `82_zone_credits.py` | Play-by-play coverage credits, ZONE coverage only (Drive+ is a zone trait). |
| `83_matched_transitions.py` | Can other DB drills measure the SAME thing as Drive+ (speed 0.8-1.0 s out of a ~45-degree break)? |
| `84_viz_data.py` | Export compact JSON for the interactive Drive+ story page (artifact). Output results/viz/drive_plus_viz.json |
| `85_build_page.py` | Build the Drive+ story page: inline the exported JSON (NaN-free) into the HTML template. |
| `85_drive_plus_page.html` | Drive+ interactive page template (v4) |
| `86_rep_film_data.py` | Film-room data: real NFL coverage reps of top-third vs bottom-third Drive+ CORNERS, by route and coverage cell. |
| `87_tailoff.py` | Why do some drill attempts tail off? Isolate them and characterize the slowdown (no ball is tracked in this drill). |
| `88_audit.py` | Audit checks not yet run on Drive+. |
| `89_wide_validation.py` | Loosened validation for thin data, applied symmetrically. |
| `90_signal_ceiling.py` | How much signal is recoverable? Attenuation, a composite criterion, and one cheap augmentation. |
| `91_within_data.py` | Improvements that stay inside the current data. |
| `92_viz_wide.py` | Page data on the registered wide sample (addenda e/f): composite criterion, zone-match framing, splits table. |
| `93_nflverse_ext.py` | Outside-data extension (nflverse play-by-play + NGS receiving aggregates), kept separate from the registered tests. |
| `94_submission_figs.py` | Submission figures on the registered wide sample, built from results/viz/drive_plus_viz_wide.json (same numbers as the page). |
| `95_build_notebook.py` | Build the Kaggle submission notebook from submission/writeup.md: markdown cells with figures embedded as data URIs, |
| `96_football_figs.py` | Football-native submission figures. |
| `97_build_nfl_page.py` | Build the Drive+ story page: inline the exported JSON (NaN-free) into the HTML template. |
| `97_nfl_page.html` | Drive+ page template, NFL-audience redesign (v5, the published page) |
| `97_nfl_page_data.py` | Data for the NFL-audience page: the wide viz JSON plus a route tree (wide reps), a mph conversion and staff-use numbers. |
| `98_cpoe.py` | CPOE instead of CP for "completion when his man is targeted" on the registered wide reps (code/89). |
| `98b_cpoe_mediation.py` | (98b) Is the harder-throw shift caused by his coverage? Copied setup from 98. |
| `99_epa_target.py` | EPA per target of his man on the registered wide reps (code/89), same estimator and standard as 89/98. |
| `data.py` | Load sequence tensors, standardize with train statistics, serve batches on device. |
| `model.py` | Masked-trajectory transformer: one context token + 80 frame tokens, reconstruct hidden kinematics. |
| `phase_utils.py` | Generic transition-phase decomposition for one combine attempt (generalizes the frozen CB drive logic). |
| `sig_utils.py` | Aggregate per-play feature matrices into player x part signatures. |
