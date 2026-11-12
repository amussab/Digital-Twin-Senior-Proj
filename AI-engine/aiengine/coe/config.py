"""Fixed design values and commissioning-time knobs for the COE pipeline.

Every default is the COE document's stated value unless the comment says ASSUMPTION.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import OPERATING_RPM, RIG_BEARING_ORDERS, RIG_HALF_WIDTH_ORDER

FS_HZ = 30_000.0            # [COE C1] CONVST rate, OS[2:0]=000
BLOCK_SETS = 256            # [COE C1] sample sets per DMA block
N_ADC_CH = 8                # AD7606 words per sample set (V7,V8 reserved)
BLOCK_MS = BLOCK_SETS / FS_HZ * 1000.0   # 8.533 ms
G = 9.80665                 # m/s2 per g


@dataclass
class CoeConfig:
    fs_hz: float = FS_HZ
    block_sets: int = BLOCK_SETS
    revolutions: int = 20                         # [COE C1]
    # --- Component 2 calibration -------------------------------------------------
    adc_range_v: float = 10.0                     # AD7606 +-10 V range pin selection (+-5 V also supported)
    accel_sens_mv_per_g: float = 100.0            # [SPEC S1] 100 mV/g
    accel_offset_v: float = 0.0                   # AC-coupled [SPEC C2]; commissioning value
    prox_sens_mv_per_um: float = 7.87             # ASSUMPTION: 200 mV/mil class Proximitor; RK4 value not in docs
    prox_offset_v: float = 0.0                    # the DC gap voltage is removed per revolution in Component 8
    # --- Component 3 / 5 -------------------------------------------------------------
    bandpass_hz: tuple = (2000.0, 8000.0)         # [COE C3]
    bandpass_order: int = 4                       # as in aiengine.dsp (4 -> 8 poles, 4 biquads)
    # [NODE FIRMWARE pipeline_config.h / envelope_coefficients.h] Butterworth order 7, 0.1 dB to 350 Hz, >= 60 dB from
    # 1250 Hz at the 30 kS/s input rate (scipy buttord(350, 1250, 0.1, 60) -> order 7, -3 dB 457.7 Hz; reproduces the
    # firmware's float32 coefficient table to 1e-8), then keep 1 sample in 6. Was order 4 at 2000 Hz in contracts <= v3b_coe.
    envelope_passband_hz: float = 350.0
    envelope_stopband_hz: float = 1250.0
    envelope_ripple_db: float = 0.1
    envelope_atten_db: float = 60.0
    decimation: int = 6                           # [COE C5]
    # --- Component 6 / 7 -------------------------------------------------------------
    fft_size: int = 4096                          # [COE C6]
    order_half_width: float = RIG_HALF_WIDTH_ORDER   # [firmware coe_project_bearing] 0.08 orders (was 0.10)
    orders: dict = field(default_factory=lambda: dict(RIG_BEARING_ORDERS))
    # --- Component 1 validity (commissioning-set in the COE doc; values here are ASSUMPTIONS) ----
    speed_policy: str = "modes"                   # "modes": near 1800/3600; "range": any rpm in rpm_range
    modes_rpm: tuple = tuple(OPERATING_RPM)       # (1800, 3600) [firmware: nominal rpm is a build setting, no tachometer]
    node_nominal_order_axis: bool = True          # firmware places the order bands with the NOMINAL rpm, not a measured one
    mode_tolerance: float = 0.01                  # +-1 % of nominal (matches aiengine.payload.is_valid)
    rpm_range: tuple = (1000.0, 3600.0)
    max_rev_deviation: float = 0.03               # each revolution within +-3 % of the window mean
    settle_samples: int = 3000                    # filters must run 100 ms before a window may start

    @property
    def envelope_rate_hz(self) -> float:
        return self.fs_hz / self.decimation

    @property
    def counts_per_volt(self) -> float:
        return 32768.0 / self.adc_range_v

    def window_samples(self, rpm: float) -> int:
        return int(round(self.revolutions * 60.0 / rpm * self.fs_hz))
