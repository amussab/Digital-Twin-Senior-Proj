"""Every assumption of the rig_anchor generator in ONE place (frozen before any model is fitted)."""
from __future__ import annotations

import os
from pathlib import Path

from ..config import ROOT

DATASET = "rig_anchor"
PROVENANCE = "[SIMULATION; endpoints resampled from rig windows, 1 bearing/fault, curve shape assumed]"
LIMITS = ("No real run-to-failure data exists for 4 of 5 fault classes. Anchors = 5 physical bearings (H-01 + one per fault); "
          "test units are held-out simulated trajectories, not held-out bearings; the damage-curve family is assumed.")

REPO = ROOT.parent.parent                                   # .../Repo
TRAIN_DIR = Path(os.environ.get("RIG_TRAINING_DATA", ROOT.parent / "COE" / "training" / "training_data"))
FAILURE_TRACE = Path(os.environ.get("RIG_FAILURE_TRACE", REPO / "rig_data" / "run_to_failure" / "run_20261115_012341.jsonl"))
LATE_PROFILE_FILE = Path(__file__).with_name("late_profile.json")      # measured from FAILURE_TRACE by `measure_late_profile`

# ---- pools (time blocks inside each real run, after the 120 s settle) ------------------------------------------------------
POOL_FRACTIONS = {"train": 0.6, "val": 0.2, "test": 0.2}   # first 60 % / next 20 % / last 20 % of each run's post-settle windows
SPLITS = ("train", "val", "test")
BLOCK_MEAN = 8.0               # mean length of a resampled contiguous block of real windows (geometric), keeps real autocorrelation
LOG_FLOOR = 1e-6

# ---- anchor placement: severity (ESTIMATED by the user on rig day, 0-3) -> u at which the MEASURED fault windows sit (g = 1) --
# ASSUMPTION. Consistent with train_real.STAGE_OF_SEV (sev3 -> stage 6 region u in [0.8,1), sev2 -> stage 4 region u in [0.4,0.6)).
U_ANCHOR = {3: 0.85, 2: 0.55}
U_ANCHOR_JITTER = 0.04         # per-unit uniform +- (stays inside the same stage region)

# ---- damage curve d(u) = a u + (1-a) u^p (synth_v3 family); train/val narrow, test WIDER (shape OOD) -------------------------
SHAPE_TRAIN = {"a": (0.20, 0.40), "p": (2.0, 3.0)}
SHAPE_TEST = {"a": (0.05, 0.60), "p": (1.5, 4.0)}
LATE_SIGMA = {"train": 0.20, "val": 0.20, "test": 0.35}   # lognormal sigma of the per-unit late-amplification scale

# ---- time axis / life scale (see reports/rig_anchor_scale_check.md) ------------------------------------------------------------
CADENCE_H = 10.0 / 60.0        # one window kept per 10-minute snapshot (as synth_v3)
RIG_C_N = 489.0                # S6700/61700 basic dynamic load rating C [CITED supplier W61700 110 lbf = 489 N; other listings 853-859 N]
RIG_PC_SIM = 0.5               # ASSUMED simulation-equivalent duty P/C (heavy end of ISO 281 validity); NOT a measured rig load
LIFE_RHO = 0.7                 # median onset->failure duration = LIFE_RHO * L_ref(rpm)  (ASSUMPTION, justified in the scale check)
LIFE_SIGMA = 0.35              # lognormal sigma of the per-unit rate/life factor
LIFE_CLIP_WINDOWS = (80, 720)
RUNIN_WINDOWS = (60, 200)      # healthy run-in before onset (fault units)
HEALTHY_UNIT_WINDOWS = (250, 700)
BASELINE_WINDOWS = 24          # legacy commissioning baseline (median of the first 24 unit windows), as the sim scoring uses.
                               # (contract policy settle 120 s + 60 s needs 180+90 windows at 1800 rpm; render_unit can prepend them)

# ---- per-unit nuisance -------------------------------------------------------------------------------------------------------
GAIN_SIGMA = 0.10              # per-channel lognormal gain on amplitude features
MOUNT_SIGMA = 0.20             # per-channel-per-feature log offset on amplitude features (same order as the observed run-to-run spread, see plausibility)
MOUNT_SIGMA_SHAPE = 0.05       # kurtosis / crest offset
DRIFT_SIGMA = 0.03             # linear sensitivity drift over the unit (fraction)

# ---- units / splits ---------------------------------------------------------------------------------------------------------
UNITS_PER_SPLIT = {"train": 240, "val": 60, "test": 60}
CLASSES = ("healthy", "inner_race", "outer_race", "ball", "cage")
FAULTS = CLASSES[1:]
SPEEDS_BY_CLASS = {"healthy": (1800, 3600), "inner_race": (1800, 3600), "outer_race": (1800, 3600), "cage": (1800, 3600),
                   "ball": (1800,)}                       # ball 3600 run jammed/broke: excluded
SPLIT_SECTION = "rig_anchor"
MASTER_SEED = 20261115
FS_HZ = 29_998.5
BAND_HZ = "2000-8000"
AMPLITUDE_IDX = [0, 3, 4, 5, 6, 7]       # within a channel's 8 features: rms, env_rms, 4 bands
SHAPE_IDX = [1, 2]                       # kurtosis, crest
