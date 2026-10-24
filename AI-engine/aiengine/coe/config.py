"""Fixed design values and commissioning-time knobs for the COE pipeline.

Every default is the COE document's stated value unless the comment says ASSUMPTION.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import RIG_BEARING_ORDERS

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
    envelope_lowpass_hz: float = 2000.0           # = envelope_rate/2.5, same as aiengine.dsp default
    envelope_lp_order: int = 4
    decimation: int = 6                           # [COE C5]
    # --- Component 6 / 7 -------------------------------------------------------------
    fft_size: int = 4096                          # [COE C6]
    order_half_width: float = 0.10                # [COE C7]
    orders: dict = field(default_factory=lambda: dict(RIG_BEARING_ORDERS))
    # --- Component 1 validity (commissioning-set in the COE doc; values here are ASSUMPTIONS) ----
    speed_policy: str = "modes"                   # "modes": near 1750/3600; "range": any rpm in rpm_range
    modes_rpm: tuple = (1750.0, 3600.0)
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
