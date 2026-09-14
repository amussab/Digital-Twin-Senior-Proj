"""Single source of truth for the ICS AI testbench.

Every constant here traces to a team document. The tracing comments matter: if a
COE or ME value changes, it changes here and nowhere else.

Sources
-------
COE  : "Project architecture/COE/COE_Processing_Pipeline_Component_Details.md"
ICS  : "Project architecture/ICS/ICS AI, Dashboard & Digital Twin Layer.md"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"


# --------------------------------------------------------------------------
# 1. Feature vector layout  [COE Component 7]
# --------------------------------------------------------------------------
# Four accelerometers, eight features each, fixed order. The COE doc requires
# this exact ordering to be shared by firmware, training data, inference code
# and dashboard labels -- so it is defined once, here.

N_ACCEL = 4
FEATURES_PER_ACCEL = 8
N_FEATURES = N_ACCEL * FEATURES_PER_ACCEL  # 32

FEATURE_NAMES_PER_ACCEL = [
    "bp_rms",       # 0 band-pass acceleration RMS        [COE C3]
    "bp_kurtosis",  # 1 band-pass acceleration kurtosis    [COE C3]
    "bp_crest",     # 2 band-pass acceleration crest factor[COE C3]
    "env_rms",      # 3 smooth-envelope RMS                [COE C5]
    "ftf_mag",      # 4 FTF-family magnitude  (cage)       [COE C6/C7]
    "bsf_mag",      # 5 BSF-family magnitude  (ball)       [COE C6/C7]
    "bpfo_mag",     # 6 BPFO-family magnitude (outer race) [COE C6/C7]
    "bpfi_mag",     # 7 BPFI-family magnitude (inner race) [COE C6/C7]
]

# Accelerometer -> measurement plane mapping [COE Component 1, ADC channel map]
#   V1 = Plane 1 X, V2 = Plane 1 Y, V3 = Plane 2 X, V4 = Plane 2 Y
ACCEL_PLANE = [1, 1, 2, 2]
ACCEL_AXIS = ["X", "Y", "X", "Y"]

FEATURE_NAMES = [
    f"a{i + 1}_{name}"
    for i in range(N_ACCEL)
    for name in FEATURE_NAMES_PER_ACCEL
]

DISPLACEMENT_NAMES = ["p1_amp_um", "p1_phase_rad", "p2_amp_um", "p2_phase_rad"]


# --------------------------------------------------------------------------
# 2. Bearing kinematics  [COE "Fixed Design Values", from the ME team]
# --------------------------------------------------------------------------
BEARING_ORDERS = {
    "FTF": 0.38,
    "BSF": 1.98,
    "BPFO": 3.05,
    "BPFI": 4.95,
}

# Steady operating modes. Transition windows are rejected upstream [COE C1].
OPERATING_RPM = (1750.0, 3600.0)
RPM_TOLERANCE = 0.01  # +/-1% steady-speed band [spec M5]

WINDOW_REVOLUTIONS = 20
def window_seconds(rpm: float) -> float:
    """Duration of one 20-revolution window [COE C1]: 1200 / RPM seconds."""
    return 1200.0 / rpm


# --------------------------------------------------------------------------
# 3. Fault taxonomy and health staging  [spec M6]
# --------------------------------------------------------------------------
# M6: "Detect/localize OD, ID, BD, CF via BCFs; failure = Miner D>=0.95 OR
#      health stage 6".  Healthy is a fifth class so the classifier can decline
#      to call a fault, which is what the Ch.2.3 false-positive analysis needs.

FAULT_CLASSES = ["healthy", "outer_race", "inner_race", "ball", "cage"]
FAULT_TO_ORDER_FAMILY = {
    "outer_race": "BPFO",
    "inner_race": "BPFI",
    "ball": "BSF",
    "cage": "FTF",
    "healthy": None,
}

N_HEALTH_STAGES = 6
FAILURE_DAMAGE = 0.95  # Miner's cumulative damage D at end of life [spec M6]

# Miner's-D boundaries separating the six health stages. Stage 1 is a long
# healthy plateau, stages compress as damage accelerates -- the standard
# accelerating shape of a cumulative-damage curve, not equal-width bins.
STAGE_DAMAGE_EDGES = [0.05, 0.20, 0.45, 0.70, 0.88]  # 5 edges -> 6 stages


# --------------------------------------------------------------------------
# 4. Specification targets this testbench checks
# --------------------------------------------------------------------------
# IDs follow the FDR's final numbering (ICS1-3 / I1-3), not the older report
# draft scheme -- see the ICS layer doc's "ID scheme note".

@dataclass(frozen=True)
class SpecTarget:
    spec_id: str
    description: str
    metric: str
    target: float
    direction: str  # "max" = measured must be <= target, "min" = >= target
    owner: str
    note: str = ""


SPEC_TARGETS: list[SpecTarget] = [
    SpecTarget(
        "ICS1", "RUL prediction accuracy (N-HiTS)",
        "rul_mape_pct", 15.0, "max", "ICS (Jaddoua)",
        note="Loosened from an earlier <=10% target; confirm which is current "
             "before quoting a number in the report.",
    ),
    SpecTarget(
        "ICS2", "Fault classification latency (TFT)",
        "tft_latency_p95_ms", 200.0, "max", "ICS (Jaddoua)",
        note="Measured per 20-revolution window, single-window batch.",
    ),
    SpecTarget(
        "I3a", "Classification performance",
        "macro_f1", 0.85, "min", "ME + COE + ICS",
        note="macro-F1 across the five fault classes.",
    ),
    SpecTarget(
        "I3b", "Health-stage estimation error",
        "stage_within_1_frac", 0.95, "min", "ME + COE + ICS",
        note="Fraction of windows whose predicted stage is within +/-1 stage. "
             "I3 states 'stage error <=1'; 0.95 is this testbench's chosen "
             "pass threshold for that, not a value from the spec table.",
    ),
    SpecTarget(
        "I3c", "Early detection",
        "caught_by_stage3_frac", 1.0, "min", "ME + COE + ICS",
        note="Fraction of faulted runs whose fault is correctly and stably "
             "called no later than health stage 3.",
    ),
    SpecTarget(
        "I1", "ICS share of end-to-end latency",
        "ics_inference_p95_ms", 500.0, "max", "ME + COE + ICS",
        note="I1's full <500ms budget spans sensor->dashboard. This measures "
             "only the ICS inference leg (both models on one window); the "
             "acquisition, DSP and network legs are COE-owned and not "
             "included here.",
    ),
]


# --------------------------------------------------------------------------
# 5. Synthetic data generation
# --------------------------------------------------------------------------
# No fault-seeded data exists yet (ME-owned dependency). Everything this
# testbench trains on is synthetic and is labelled as such in every report it
# writes. These figures are plausible-by-construction, not measured.

@dataclass
class SynthConfig:
    n_runs: int = 48                  # independent run-to-failure histories
    min_windows: int = 220            # shortest run, in stored windows
    max_windows: int = 620            # longest run, in stored windows
    # Machine-time between two *stored* windows. The node emits a window every
    # 0.33-0.69 s, but the RUL path consumes a decimated stream; one stored
    # window per 10 minutes of machine life is the assumed monitoring cadence.
    minutes_per_window: float = 10.0
    healthy_fraction: float = 0.18    # share of runs that never develop a fault
    seed: int = 20260914

    @property
    def hours_per_window(self) -> float:
        return self.minutes_per_window / 60.0


# --------------------------------------------------------------------------
# 6. Model hyperparameters
# --------------------------------------------------------------------------
# Deliberately small. This is a spec-verification testbench meant to run on the
# candidate hardware (a laptop or a Pi 5 -- still unresolved, ICS doc SS6), not
# a maximum-accuracy training run.

@dataclass
class NHiTSConfig:
    """N-HiTS forecasts the health index forward; RUL is read off the
    threshold crossing. See rul.py for why it is not trained on RUL directly."""
    encoder_length: int = 48          # 8 h of machine time at 10 min/window
    prediction_length: int = 12       # 2 h forecast horizon
    hidden_size: int = 64
    n_blocks: tuple = (1, 1, 1)
    learning_rate: float = 3e-3
    batch_size: int = 128
    max_epochs: int = 30
    dropout: float = 0.1


@dataclass
class TFTConfig:
    """TFT classifies one window at a time (decoder length 1)."""
    encoder_length: int = 24          # 4 h of context at 10 min/window
    prediction_length: int = 1        # per-window classification [ICS2]
    hidden_size: int = 32
    attention_head_size: int = 4
    hidden_continuous_size: int = 16
    learning_rate: float = 2e-3
    batch_size: int = 128
    max_epochs: int = 30
    dropout: float = 0.15


@dataclass
class RunConfig:
    synth: SynthConfig = field(default_factory=SynthConfig)
    nhits: NHiTSConfig = field(default_factory=NHiTSConfig)
    tft: TFTConfig = field(default_factory=TFTConfig)
    seed: int = 20260914
    # Fraction of runs held out. Splitting is by *run*, never by window --
    # windows from one bearing's history are not independent samples.
    val_fraction: float = 0.15
    test_fraction: float = 0.20
    latency_windows: int = 200        # single-window inferences to time
