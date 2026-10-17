# Tier A pre-registration
written 2026-10-11T23:39:56 before computing any tier A statistic

| id | unit | combine feature | NFL outcome | expected | rationale |
|---|---|---|---|---|---|
| A1 | WR | `fam:routes:accel_0_1s` | REC sep_oe vs man | + | acceleration into the route beats man coverage |
| A2 | WR | `fam:routes:brk1_entry_speed` | REC sep_oe in-breaking | + | speed carried into the break |
| A3 | WR | `drill:WR:SPEED_OUT_ROUTE_LEFT:turn_speed` | REC sep_oe out-breaking | + | speed through the out-cut |
| A4 | WR | `drill:THREE_CONE_DRILL:brk1_exit_burst` | COV created sep_gain_break | + | burst out of a plant opens the receiver at his break |
| A5 | WR | `fam:routes:catch_carry` | REC YAC over NGS expected | + | carrying speed through the catch |
| A6 | WR | `drill:FORTY_YARD_DASH:top_speed` | REC sep_oe vertical | + | top speed on go routes |
| A7 | WR | `drill:WR:CURL_ROUTE_RIGHT:turn_lean` | REC sep_oe stop | + | sharp curl turn on stop routes |
| B1 | CB | `fam:backpedal:speed_at_1.5s` | COV allowed (man) sep_mean | - | faster backpedal stays on top in man |
| B2 | CB | `drill:DB:W_DRILL:brk_mean_exit_burst` | COV allowed sep_gain_break | - | burst out of breaks limits what the receiver gains at his break |
| B3 | CB | `drill:FORTY_YARD_DASH:top_speed` | COV allowed db_speed_deficit | - | top speed closes the speed gap after the break |
| B4 | CB | `fam:all_db:speed_at_0.5s` | TACKLE closing_speed | + | first-step burst closes on the ball carrier |
| B5 | CB | `drill:DB:W_DRILL:brk_mean_brake_time` | TACKLE yds_after_arrival | + | slow braking -> overrun -> more yards after arrival |
| C1 | S | `fam:all_db:speed_at_0.5s` | TACKLE closing_speed | + | first-step burst closes on the ball carrier |
| C2 | S | `drill:FORTY_YARD_DASH:top_speed` | S run tackles_oe | + | range in run support |
| D1 | EDGE | `frozen:hoop_turn_speed` | EDGE penetration_oe | + | bend at speed (pre-registered bend metric) |
| D2 | EDGE | `drill:DL:PASS_RUSH_DRILL:turn_speed` | H2H corner rate | + | turn speed around the bag -> turning the tackle's corner |
| D3 | EDGE | `fam:rush:accel_sustained_peak` | EDGE get_off_oe (lower=better) | - | sustained acceleration -> faster get-off |
| D4 | EDGE | `drill:FORTY_YARD_DASH:accel_sustained_peak` | EDGE depth2s_oe | + | acceleration -> depth at 2 s |
| E1 | IDL | `frozen:hoop_turn_speed` | IDL penetration_oe | + | bend at speed |
| E2 | IDL | `drill:THREE_CONE_DRILL:accel_0_1s` | IDL get_off_oe (lower=better) | - | short-area acceleration -> get-off |
| E3 | IDL | `fam:all_dl:turn_speed` | H2H blocker ppa_oe vs him | + | turn speed -> wins vs guards / centers |
| F1 | T | `drill:OL:PASS_PRO_MIRROR_DRILL:brk_mean_entry_speed` | T pressure prob allowed_oe (lower=better) | - | mirror speed -> fewer pressures |
| F2 | T | `drill:OL:FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE:turn_lean` | H2H corner allowed rate | - | redirect in the wave drill -> fewer corners turned |
| G1 | IOL | `drill:OL:PASS_PRO_MIRROR_DRILL:brk_mean_entry_speed` | IOL pressure prob allowed_oe (lower=better) | - | mirror speed -> fewer pressures |
| G2 | IOL | `fam:lateral:top_speed` | H2H ppa_oe | - | lateral speed -> lower pressure probability in head-to-head reps |

Rep-level:
- R1 (coverage, outcome sep_gain_break): WR drill:THREE_CONE_DRILL:brk1_exit_burst (+) x DB drill:DB:W_DRILL:brk_mean_exit_burst (-); interaction < 0 (receiver burst pays most vs low-burst DBs)
- R2 (coverage (man), outcome sep_mean): WR fam:routes:accel_0_1s (+), DB fam:backpedal:speed_at_1.5s (-)
- R3 (tackle, outcome yds_after_arrival): tackler fam:all_db:speed_at_0.5s (-) [DB tacklers]
