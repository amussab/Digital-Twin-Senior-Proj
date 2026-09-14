"""Adapters that bring public bearing datasets into this project's feature space.

This module answers risk 1 from `ICS External Datasets & Literature Precedent.md`:

    "CWRU/FEMTO-ST/XJTU-SY are raw vibration signals from different bearings at
     different sample rates -- they are not usable as pretraining input until
     they're run through the *same* feature-extraction pipeline the rig's own
     data will use. That reprocessing step is real engineering work and isn't
     currently scoped anywhere in the project plan."

Each adapter takes a dataset's raw signals and produces the identical schema
`synth.py` produces, so anything downstream -- feature engineering, both
models, the spec harness -- is unchanged regardless of the source. Three things
have to be handled per dataset, and getting any of them wrong silently
poisons the pretraining rather than failing:

1.  SAMPLE RATE. Passed to dsp.PipelineConfig, which narrows the resonance band
    if the dataset's Nyquist cannot carry 2-8 kHz and sets `band_was_narrowed`
    so the mismatch is visible rather than assumed away.
2.  BEARING GEOMETRY. Every dataset uses a different bearing, so the order
    bands Component 7 integrates over differ. Using this project's 6200-class
    orders on CWRU's bearing would integrate energy at frequencies where
    CWRU's bearing has no defect signature.
3.  CHANNEL COUNT. FEMTO-ST and XJTU-SY have two accelerometers on one plane;
    this project has four across two planes. See `ChannelPolicy` -- the default
    refuses rather than inventing the missing plane.

RUL labels are emitted BOTH as absolute hours and as percent-of-life-remaining,
which is risk 2's recommendation: absolute hours are not comparable between a
FEMTO accelerated test and this project's ISO-281-style life law, but life
fraction is.

NOTHING HERE DOWNLOADS ANYTHING. Each loader looks for data already on disk
under `AI-test/data/external/<name>/` and, if it is absent, prints exactly
where to get it -- using the corrected URLs from the literature doc, since two
of the originally-supplied mirrors pointed at the wrong datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

import dsp
from config import DATA_DIR, FAULT_CLASSES

EXTERNAL_DIR = DATA_DIR / "external"


class ChannelPolicy(str, Enum):
    """How to handle datasets with fewer accelerometers than the rig.

    STRICT is the default and refuses to proceed. The alternative, mirroring
    one plane onto the other, fabricates a second measurement plane that was
    never instrumented -- the model would learn "plane asymmetry is always
    zero" from it, which is false on the real rig and would suppress exactly
    the feature that localises a fault to one bearing.
    """

    STRICT = "strict"            # refuse; the honest default
    MIRROR_PLANE = "mirror"      # duplicate plane 1 onto plane 2 -- smoke-test only


@dataclass
class DatasetSpec:
    """Everything needed to bring one public dataset into this feature space."""

    name: str
    sample_rate_hz: float
    geometry: dsp.BearingGeometry
    n_channels: int
    has_run_to_failure: bool
    has_fault_labels: bool
    nominal_rpm: dict[str, float]
    source_urls: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def directory(self) -> Path:
        return EXTERNAL_DIR / self.name

    def pipeline(self, rpm: float) -> dsp.PipelineConfig:
        return dsp.PipelineConfig(
            sample_rate_hz=self.sample_rate_hz,
            bearing_orders=dsp.bearing_orders_from_geometry(self.geometry),
        )


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# Geometry figures below are the published values for each rig's bearing and
# MUST be re-checked against the dataset's own documentation before any
# pretraining run is reported. They set the order bands, so an error here
# quietly extracts the wrong frequencies rather than raising anything.

FEMTO = DatasetSpec(
    name="femto",
    sample_rate_hz=25_600.0,
    geometry=dsp.BearingGeometry(
        n_rolling_elements=13, ball_diameter_mm=3.5, pitch_diameter_mm=25.6,
        contact_angle_deg=0.0, name="PRONOSTIA test bearing",
    ),
    n_channels=2,                       # horizontal + vertical, one plane
    has_run_to_failure=True,
    has_fault_labels=False,             # degradation, not a seeded fault type
    nominal_rpm={"Bearing1": 1800.0, "Bearing2": 1650.0, "Bearing3": 1500.0},
    source_urls=[
        "https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset",
        "https://publiweb.femto-st.fr/tntnet/entries/1528/documents/author/data",
    ],
    notes="17 bearings, 3 conditions, lifespans ~1-7 h. The only dataset here "
          "with genuine time-to-failure labels, so this is ICS1's pretraining "
          "source. Do NOT use the ieee-dataport.org/node/1849 link -- it "
          "resolves to an unrelated PHM-2009 listing.",
)

CWRU = DatasetSpec(
    name="cwru",
    sample_rate_hz=12_000.0,            # drive-end 12k; 48k variant also exists
    geometry=dsp.BearingGeometry(
        n_rolling_elements=9, ball_diameter_mm=7.94, pitch_diameter_mm=39.04,
        contact_angle_deg=0.0, name="SKF 6205-2RS JEM drive end",
    ),
    n_channels=2,                       # drive-end + fan-end, different planes
    has_run_to_failure=False,           # seeded faults, no degradation history
    has_fault_labels=True,
    nominal_rpm={"0hp": 1797.0, "1hp": 1772.0, "2hp": 1750.0, "3hp": 1730.0},
    source_urls=[
        "https://engineering.case.edu/bearingdatacenter/download-data-file",
    ],
    notes="Seeded faults at set diameters -- trains ICS2's classifier, not "
          "ICS1's RUL path, because there is no run-to-failure history. "
          "`pip install cwru` exists but has known download-URL breakage; "
          "smoke-test it rather than assuming it works unmodified.",
)

XJTU_SY = DatasetSpec(
    name="xjtu_sy",
    sample_rate_hz=25_600.0,
    geometry=dsp.BearingGeometry(
        n_rolling_elements=8, ball_diameter_mm=7.92, pitch_diameter_mm=34.55,
        contact_angle_deg=0.0, name="LDK UER204",
    ),
    n_channels=2,                       # horizontal + vertical, one plane
    has_run_to_failure=True,
    has_fault_labels=True,
    nominal_rpm={"35Hz12kN": 2100.0, "37.5Hz11kN": 2250.0, "40Hz10kN": 2400.0},
    source_urls=[
        "https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets",
        "https://data.mendeley.com/datasets/mpn45f4gxc",
    ],
    notes="15 bearings, 3 conditions. ICS1 cross-validation -- checks the "
          "model is not overfit to FEMTO's particular rig. The "
          "cathysiyu/Mechanical-datasets mirror does NOT contain this data.",
)

REGISTRY = {spec.name: spec for spec in (FEMTO, CWRU, XJTU_SY)}


# ---------------------------------------------------------------------------
# Channel mapping
# ---------------------------------------------------------------------------

def map_channels(channels: list[np.ndarray], spec: DatasetSpec,
                 policy: ChannelPolicy) -> list[np.ndarray]:
    """Fit a dataset's channels into the project's four-channel layout.

    Project layout [COE Component 1]: a1 = plane 1 X, a2 = plane 1 Y,
    a3 = plane 2 X, a4 = plane 2 Y.
    """
    if len(channels) >= 4:
        return channels[:4]
    if policy is ChannelPolicy.STRICT:
        raise ValueError(
            f"{spec.name} has {len(channels)} accelerometer channel(s); this "
            f"project's feature vector assumes 4 across 2 measurement planes. "
            f"Options, in order of honesty:\n"
            f"  1. Train a 2-channel variant (16 features) for pretraining, "
            f"then extend to 4 channels when fine-tuning on the rig. This is "
            f"the recommended path -- it changes no data.\n"
            f"  2. Drop the plane-asymmetry feature and pretrain on "
            f"per-channel features only.\n"
            f"  3. ChannelPolicy.MIRROR_PLANE -- duplicates plane 1 onto "
            f"plane 2. This FABRICATES a measurement plane and teaches the "
            f"model that plane asymmetry is always zero. Smoke-testing only; "
            f"never for a reported number."
        )
    # MIRROR_PLANE: explicitly requested, explicitly fabricated.
    base = list(channels)
    while len(base) < 4:
        base.append(channels[len(base) % len(channels)])
    return base[:4]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _missing(spec: DatasetSpec) -> None:
    lines = [
        f"\n{spec.name.upper()} is not present at {spec.directory}.",
        "",
        "This module never downloads anything. Fetch it manually from:",
    ]
    lines += [f"  - {url}" for url in spec.source_urls]
    lines += ["", f"Then place it under {spec.directory}/", "", spec.notes, ""]
    print("\n".join(lines))


def load_femto(policy: ChannelPolicy = ChannelPolicy.STRICT,
               spec: DatasetSpec = FEMTO) -> pd.DataFrame | None:
    """FEMTO-ST / PRONOSTIA run-to-failure -> this project's window schema.

    Expected layout: <directory>/<BearingX_Y>/acc_*.csv, each CSV one 0.1 s
    snapshot of 2560 samples with columns
    [hour, minute, second, microsecond, horizontal, vertical], snapshots taken
    every 10 s. One snapshot becomes one window: 0.1 s at 1800 rpm is 3
    revolutions, not the project's 20, which is itself a mismatch worth
    stating -- the order resolution is coarser than the rig's 0.05 order.
    """
    if not spec.directory.exists():
        _missing(spec)
        return None

    rows = []
    for bearing_dir in sorted(p for p in spec.directory.iterdir() if p.is_dir()):
        files = sorted(bearing_dir.glob("acc_*.csv"))
        if not files:
            continue
        condition = bearing_dir.name.split("_")[0]
        rpm = spec.nominal_rpm.get(condition, 1800.0)
        cfg = spec.pipeline(rpm)
        total = len(files)

        for index, path in enumerate(files):
            raw = pd.read_csv(path, header=None).to_numpy()
            if raw.shape[1] < 6:
                continue
            channels = [raw[:, 4].astype(float), raw[:, 5].astype(float)]
            channels = map_channels(channels, spec, policy)
            features = dsp.window_features(channels, rpm, cfg)

            # Snapshots are 10 s apart; RUL is the time to the last snapshot.
            hours_elapsed = index * 10.0 / 3600.0
            rul_hours = (total - 1 - index) * 10.0 / 3600.0
            rows.append(
                {
                    "run_id": bearing_dir.name,
                    "window_index": index,
                    "hours_elapsed": hours_elapsed,
                    "rpm": rpm,
                    "operating_mode": condition,
                    "fault_class": "unlabelled",
                    "rul_hours": rul_hours,
                    # Risk 2: the scale-invariant label. Comparable across
                    # datasets and across life laws; absolute hours are not.
                    "life_fraction_remaining": (total - 1 - index) / max(total - 1, 1),
                    "band_was_narrowed": cfg.band_was_narrowed,
                    **dict(zip(_feature_columns(), features)),
                }
            )
        print(f"  {bearing_dir.name}: {total} windows, {rpm:.0f} rpm")

    return pd.DataFrame(rows) if rows else None


def load_cwru(policy: ChannelPolicy = ChannelPolicy.STRICT,
              spec: DatasetSpec = CWRU) -> pd.DataFrame | None:
    """CWRU seeded-fault data -> this project's window schema.

    Expected layout: <directory>/*.mat, MATLAB files as distributed, with
    variable names containing DE_time / FE_time and (usually) RPM. Files are
    segmented into 20-revolution windows at the file's own speed, so a single
    long recording yields many windows -- which is what the classifier needs
    and is the only thing CWRU can supply, since it has no degradation history.
    """
    if not spec.directory.exists():
        _missing(spec)
        return None
    try:
        from scipy.io import loadmat
    except ImportError:
        print("  scipy.io is required to read CWRU .mat files.")
        return None

    rows = []
    for path in sorted(spec.directory.glob("*.mat")):
        contents = loadmat(path)
        drive = [v for k, v in contents.items() if k.endswith("DE_time")]
        fan = [v for k, v in contents.items() if k.endswith("FE_time")]
        if not drive:
            continue
        rpm_entry = [v for k, v in contents.items() if k.endswith("RPM")]
        rpm = float(np.ravel(rpm_entry[0])[0]) if rpm_entry else 1750.0
        cfg = spec.pipeline(rpm)

        signals = [np.ravel(drive[0]).astype(float)]
        if fan:
            signals.append(np.ravel(fan[0]).astype(float))
        signals = map_channels(signals, spec, policy)

        fault_class = _cwru_label(path.name)
        span = dsp.samples_per_window(rpm, cfg)
        n_windows = len(signals[0]) // span
        for index in range(n_windows):
            slice_ = slice(index * span, (index + 1) * span)
            features = dsp.window_features([s[slice_] for s in signals], rpm, cfg)
            rows.append(
                {
                    "run_id": path.stem,
                    "window_index": index,
                    "hours_elapsed": index * span / cfg.sample_rate_hz / 3600.0,
                    "rpm": rpm,
                    "operating_mode": f"{rpm:.0f}",
                    "fault_class": fault_class,
                    "rul_hours": np.nan,     # seeded fault: no time-to-failure
                    "life_fraction_remaining": np.nan,
                    "band_was_narrowed": cfg.band_was_narrowed,
                    **dict(zip(_feature_columns(), features)),
                }
            )
        print(f"  {path.name}: {n_windows} windows, {rpm:.0f} rpm, {fault_class}")

    return pd.DataFrame(rows) if rows else None


def _cwru_label(filename: str) -> str:
    """Map a CWRU filename to this project's fault taxonomy.

    CWRU has no cage-fault class, so `cage` is simply absent from CWRU
    pretraining -- worth knowing, because a classifier pretrained on CWRU has
    seen no cage examples at all and depends entirely on the fine-tuning set
    for that class.
    """
    name = filename.upper()
    if name.startswith("IR"):
        return "inner_race"
    if name.startswith("OR"):
        return "outer_race"
    if name.startswith("B") and not name.startswith("BA"):
        return "ball"
    if "NORMAL" in name or name.startswith("N"):
        return "healthy"
    return "unlabelled"


def _feature_columns() -> list[str]:
    from config import FEATURE_NAMES

    return FEATURE_NAMES


LOADERS = {"femto": load_femto, "cwru": load_cwru}


def load(name: str, policy: ChannelPolicy = ChannelPolicy.STRICT
         ) -> pd.DataFrame | None:
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(REGISTRY)}")
    if name not in LOADERS:
        spec = REGISTRY[name]
        print(f"\n{name} has a spec but no loader yet. Its layout is close to "
              f"FEMTO's (two channels, {spec.sample_rate_hz:.0f} Hz); adapt "
              f"load_femto once the data is on disk.")
        _missing(spec)
        return None
    print(f"\n[{name}] loading from {REGISTRY[name].directory}")
    return LOADERS[name](policy)


def describe() -> str:
    """Human-readable summary of what each dataset can and cannot support."""
    lines = ["Public datasets registered for pretraining", "=" * 62]
    for spec in REGISTRY.values():
        orders = dsp.bearing_orders_from_geometry(spec.geometry)
        present = "PRESENT" if spec.directory.exists() else "not downloaded"
        lines += [
            "",
            f"{spec.name.upper()}  [{present}]  -- {spec.geometry.name}",
            f"  sample rate      : {spec.sample_rate_hz:,.0f} Hz",
            f"  channels         : {spec.n_channels} (project expects 4)",
            f"  run-to-failure   : {'yes -- usable for ICS1 RUL' if spec.has_run_to_failure else 'no -- ICS2 classification only'}",
            f"  fault labels     : {'yes' if spec.has_fault_labels else 'no'}",
            "  bearing orders   : " + ", ".join(
                f"{k} {v:.2f}" for k, v in orders.items()
            ),
            f"  get it from      : {spec.source_urls[0]}",
        ]
    return "\n".join(lines)
