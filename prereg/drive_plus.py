"""FROZEN (2026-10-12): Drive+ -- the headline grade of CB Transition Drive Speed.

Do not edit. PREREG_cb_drive_speed_addendum_2026-10-12d.md records this file's SHA-256.

  Drive+ = 100 + 10 * (z - MU_CB) / SD_CB
    raw    = cb_drive_speed() from the frozen file cb_drive_speed.py (SHA-256 009e06e2...), i.e. the player's mean drive
             top speed (yd/s) over his valid attempts of BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION.
    z      = raw z-scored WITHIN the combine class (draft_year) over every player in that class with a valid raw value
             (sample SD, ddof = 1). Within-class scaling absorbs year-to-year differences in clip length and setup.
    MU_CB, SD_CB = mean and SD of z over the 59 pool CBs of the 2023-25 classes (fixed constants below).
  So 100 = the average 2023-25 pool CB and 10 points = one SD of those CBs.
  Tiers: "110+" (>= 110), "100-110", "90-100", "<90".
"""
import numpy as np
import pandas as pd
from cb_drive_speed import cb_drive_speed

MU_CB = 0.07941543622443174
SD_CB = 1.0405347113038825
TIERS = [(110, np.inf, "110+"), (100, 110, "100-110"), (90, 100, "90-100"), (-np.inf, 90, "<90")]


def drive_plus(combine: pd.DataFrame) -> pd.DataFrame:
    raw = cb_drive_speed(combine)
    cls = combine.drop_duplicates("nfl_id").set_index("nfl_id").draft_year.reindex(raw.index)
    z = (raw - raw.groupby(cls).transform("mean")) / raw.groupby(cls).transform("std")
    dp = 100 + 10 * (z - MU_CB) / SD_CB
    tier = dp.map(lambda v: next(lab for lo, hi, lab in TIERS if lo <= v < hi))
    return pd.DataFrame({"cb_drive_speed": raw, "draft_year": cls, "drive_plus": dp, "tier": tier})
