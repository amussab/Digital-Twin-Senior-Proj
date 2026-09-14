"""COE's Components 3-7 feature pipeline, re-implemented in Python.

Why this module exists
----------------------
The external-datasets doc (risk 1) states the problem exactly: N-HiTS and TFT
are fed this project's specific 32-value envelope + order-FFT feature set,
computed for this project's bearing geometry and window scheme, while CWRU,
FEMTO-ST and XJTU-SY are raw vibration at different sample rates from different
bearings. Those datasets "are not usable as pretraining input until they're run
through the *same* feature-extraction pipeline the rig's own data will use",
and that reprocessing step was not scoped anywhere.

This module is that step. It mirrors COE Components 3 through 7 exactly, but
parameterised by sample rate, passband and bearing geometry, so the same code
turns the team's own rig data, CWRU, FEMTO-ST or XJTU-SY into one identical
32-value feature contract. Whatever the models are pretrained on then lives in
the same feature space as the rig data they are fine-tuned on.

It also serves a second purpose: it is an independent implementation of COE's
spec. Where it disagrees with the STM32 firmware, one of the two is wrong, and
finding that out on a bench is much cheaper than finding it out in integration.

Pipeline, following COE Components 3-7
--------------------------------------
    C3  band-pass to the bearing-resonance band, accumulate RMS/kurtosis/crest
    C4  full-wave rectify
    C5  envelope low-pass, then decimate to the envelope rate
    C6  remove mean, Hann window, zero-pad to 4096, rFFT, map bins to orders
    C7  eight features per channel, in COE's fixed order
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from config import BEARING_ORDERS, FEATURE_NAMES_PER_ACCEL


@dataclass(frozen=True)
class BearingGeometry:
    """Physical bearing description, for deriving characteristic orders."""

    n_rolling_elements: int
    ball_diameter_mm: float
    pitch_diameter_mm: float
    contact_angle_deg: float = 0.0
    name: str = "unnamed"


def bearing_orders_from_geometry(geometry: BearingGeometry) -> dict[str, float]:
    """Characteristic bearing-defect orders from geometry.

    Standard kinematic relations, in shaft orders (multiples of shaft speed):

        FTF  = (1/2)  (1 - (d/D) cos a)
        BPFO = (N/2)  (1 - (d/D) cos a)
        BPFI = (N/2)  (1 + (d/D) cos a)
        BSF  = (D/2d) (1 - ((d/D) cos a)^2)

    Needed because every external dataset uses a different bearing, and the
    order bands Component 7 integrates over are only meaningful for the
    bearing actually installed. Feeding CWRU data through the project's own
    6200-class orders would extract energy at frequencies where CWRU's bearing
    has no defect signature at all.
    """
    ratio = geometry.ball_diameter_mm / geometry.pitch_diameter_mm
    cos_alpha = np.cos(np.deg2rad(geometry.contact_angle_deg))
    x = ratio * cos_alpha
    n = geometry.n_rolling_elements
    return {
        "FTF": 0.5 * (1.0 - x),
        "BSF": (1.0 / (2.0 * ratio)) * (1.0 - x**2),
        "BPFO": (n / 2.0) * (1.0 - x),
        "BPFI": (n / 2.0) * (1.0 + x),
    }


@dataclass
class PipelineConfig:
    """Everything Components 3-7 need that is not the signal itself."""

    sample_rate_hz: float
    bandpass_hz: tuple[float, float] = (2000.0, 8000.0)   # COE C3 baseline
    envelope_rate_hz: float = 5000.0                      # COE C5
    fft_size: int = 4096                                  # COE C6
    order_half_width: float = 0.10                        # COE C7
    bearing_orders: dict[str, float] = field(
        default_factory=lambda: dict(BEARING_ORDERS)
    )
    bandpass_order: int = 4
    envelope_lowpass_hz: float | None = None              # default: Nyquist/1.25
    # Set when the requested resonance band did not fit under the dataset's
    # Nyquist limit and had to be narrowed. Features computed with a narrowed
    # band are NOT directly comparable with the rig's own.
    band_was_narrowed: bool = False

    def __post_init__(self) -> None:
        nyquist = self.sample_rate_hz / 2.0
        low, high = self.bandpass_hz
        if high >= nyquist:
            # A dataset sampled too slowly to contain the project's 2-8 kHz
            # resonance band cannot simply be run with the band clipped: the
            # features would mean something different. Narrow it and say so.
            new_high = 0.95 * nyquist
            if low >= new_high:
                raise ValueError(
                    f"sample rate {self.sample_rate_hz:.0f} Hz cannot support "
                    f"any part of the {low:.0f}-{high:.0f} Hz resonance band "
                    f"(Nyquist {nyquist:.0f} Hz). This dataset is not usable "
                    f"with the project's band without redefining it."
                )
            self.bandpass_hz = (low, new_high)
            self.band_was_narrowed = True
        if self.envelope_rate_hz > self.sample_rate_hz:
            self.envelope_rate_hz = self.sample_rate_hz
        if self.envelope_lowpass_hz is None:
            self.envelope_lowpass_hz = self.envelope_rate_hz / 2.5

    @property
    def decimation_factor(self) -> int:
        return max(1, int(round(self.sample_rate_hz / self.envelope_rate_hz)))


def bandpass_filter(x: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Component 3: keep the bearing-resonance band."""
    nyquist = cfg.sample_rate_hz / 2.0
    low = max(cfg.bandpass_hz[0] / nyquist, 1e-4)
    high = min(cfg.bandpass_hz[1] / nyquist, 0.999)
    sos = signal.butter(cfg.bandpass_order, [low, high], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, x)


def envelope(x_bandpassed: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Components 4 and 5: rectify, low-pass, decimate.

    Rectification then low-pass, rather than a Hilbert transform, because that
    is what COE Component 4/5 specifies and what the STM32 will actually run.
    Using a mathematically nicer envelope here would make these features
    subtly incomparable with the firmware's.
    """
    rectified = np.abs(x_bandpassed)
    nyquist = cfg.sample_rate_hz / 2.0
    cutoff = min(cfg.envelope_lowpass_hz / nyquist, 0.999)
    sos = signal.butter(4, cutoff, btype="lowpass", output="sos")
    smoothed = signal.sosfiltfilt(sos, rectified)
    return smoothed[:: cfg.decimation_factor]


def envelope_spectrum(env: np.ndarray, rpm: float, cfg: PipelineConfig
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Component 6: envelope spectrum with an order axis.

    Returns (orders, magnitudes). Zero-padding to a fixed FFT size gives a
    denser grid, not more resolution -- the physical resolution is set by the
    record length, as COE's own note says.
    """
    record = env - env.mean()
    if len(record) < 8:
        return np.zeros(0), np.zeros(0)
    windowed = record * np.hanning(len(record))
    padded = np.zeros(max(cfg.fft_size, len(windowed)))
    padded[: len(windowed)] = windowed

    spectrum = np.fft.rfft(padded)
    magnitudes = np.abs(spectrum) * 2.0 / len(windowed)
    envelope_fs = cfg.sample_rate_hz / cfg.decimation_factor
    frequencies = np.fft.rfftfreq(len(padded), d=1.0 / envelope_fs)
    shaft_hz = rpm / 60.0
    if shaft_hz <= 0:
        return np.zeros(0), np.zeros(0)
    return frequencies / shaft_hz, magnitudes


def order_band_magnitude(orders: np.ndarray, magnitudes: np.ndarray,
                         centre_order: float, half_width: float) -> float:
    """Component 7: combined magnitude in a narrow band around one order."""
    if orders.size == 0:
        return 0.0
    selected = (orders >= centre_order - half_width) & (
        orders <= centre_order + half_width
    )
    if not selected.any():
        return 0.0
    return float(np.sqrt(np.sum(magnitudes[selected] ** 2)))


def channel_features(x: np.ndarray, rpm: float, cfg: PipelineConfig) -> np.ndarray:
    """The eight features Component 7 produces for one accelerometer.

    Order matches config.FEATURE_NAMES_PER_ACCEL exactly:
        bp_rms, bp_kurtosis, bp_crest, env_rms,
        ftf_mag, bsf_mag, bpfo_mag, bpfi_mag
    """
    filtered = bandpass_filter(np.asarray(x, dtype=np.float64), cfg)

    rms = float(np.sqrt(np.mean(filtered**2)))
    if rms <= 0 or not np.isfinite(rms):
        return np.zeros(len(FEATURE_NAMES_PER_ACCEL), dtype=np.float32)
    centred = filtered - filtered.mean()
    variance = float(np.mean(centred**2))
    kurtosis = float(np.mean(centred**4) / variance**2) if variance > 0 else 0.0
    crest = float(np.max(np.abs(filtered)) / rms)

    env = envelope(filtered, cfg)
    env_rms = float(np.sqrt(np.mean(env**2)))

    orders, magnitudes = envelope_spectrum(env, rpm, cfg)
    families = [
        order_band_magnitude(orders, magnitudes, cfg.bearing_orders[family],
                             cfg.order_half_width)
        for family in ("FTF", "BSF", "BPFO", "BPFI")
    ]
    return np.asarray([rms, kurtosis, crest, env_rms, *families], dtype=np.float32)


def window_features(channels: list[np.ndarray], rpm: float,
                    cfg: PipelineConfig) -> np.ndarray:
    """The full 32-value vector for one window, given four channels.

    Fewer than four channels is an error, not something to pad around: every
    external dataset here has fewer accelerometers than the project's rig, and
    silently duplicating or zero-filling the missing ones would fabricate data
    the model would then learn from. external.py decides how to handle that
    gap explicitly.
    """
    if len(channels) != 4:
        raise ValueError(
            f"expected 4 accelerometer channels, got {len(channels)}. "
            f"See external.py's channel-mapping policy -- do not pad here."
        )
    return np.concatenate([channel_features(c, rpm, cfg) for c in channels])


def samples_per_window(rpm: float, cfg: PipelineConfig,
                       revolutions: int = 20) -> int:
    """Samples spanning one 20-revolution window at a given speed [COE C1]."""
    return int(round(revolutions * 60.0 / rpm * cfg.sample_rate_hz))
