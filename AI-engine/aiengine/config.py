"""Constants mirrored from team documents. Change the doc first, then this file.

Sources
-------
COE  : Project architecture/COE/COE_Processing_Pipeline_Component_Details.md (main, 2026-09-17)
SPEC : Project architecture/overview/Specifications.md (copy of the team xlsx)
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]           # AI-engine/
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
CHECKPOINT_DIR = ROOT / "checkpoints"

# --- COE feature contract [COE C7] -----------------------------------------
FEATURE_NAMES_PER_ACCEL = [
    "bp_rms", "bp_kurtosis", "bp_crest", "env_rms",
    "ftf_mag", "bsf_mag", "bpfo_mag", "bpfi_mag",
]
AXES = ("x", "y")
BEARING_FEATURES = [f"{a}_{n}" for a in AXES for n in FEATURE_NAMES_PER_ACCEL]  # 16
N_BEARINGS_RIG = 2
# Rig payload slice per bearing: bearing 1 -> [0:16], bearing 2 -> [16:32]
RIG_BEARING_SLICES = {1: slice(0, 16), 2: slice(16, 32)}

# Rig bearing characteristic orders [COE "Fixed Design Values", from ME]
RIG_BEARING_ORDERS = {"FTF": 0.38, "BSF": 1.98, "BPFO": 3.05, "BPFI": 4.95}

RIG_SAMPLE_RATE_HZ = 30_000.0          # [COE C1] AD7606, OS disabled
BANDPASS_HZ = (2000.0, 8000.0)         # [COE C3]
ENVELOPE_RATE_HZ = 5000.0              # [COE C5]
FFT_SIZE = 4096                        # [COE C6]
WINDOW_REVOLUTIONS = 20                # [COE C1]
OPERATING_RPM = (1750.0, 3600.0)       # [COE] steady modes
RPM_TOLERANCE = 0.01                   # +/-1% steady-speed band [SPEC M5, via AI-test/config.py]

# --- Fault taxonomy [SPEC IS3] ----------------------------------------------
FAULT_CLASSES = ["healthy", "outer_race", "inner_race", "ball", "cage"]
N_HEALTH_STAGES = 6

# --- Spec targets [SPEC] ------------------------------------------------------
SPEC_TARGETS = {
    "S7_rul_mape_pct": 15.0,          # <=
    "S8_tft_latency_ms": 200.0,       # <
    "IS1_e2e_latency_ms": 500.0,      # <
    "IS3_macro_f1": 0.85,             # >=
    "IS3_stage_error_max": 1,         # <=
    "IS3_caught_by_stage": 3,         # <=
    "IS2_residual_pct": 5.0,          # <=
}

SEED = 20261004
