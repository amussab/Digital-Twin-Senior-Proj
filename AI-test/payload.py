"""Codec for COE's fixed application payload.

Layout [COE Component 9], little-endian throughout:

    offset  type        field
    0-3     Float32     average RPM over the 20-revolution window
    4-131   Float32[32] ordered accelerometer features
    132-135 Float32     plane 1 radial amplitude, micrometres peak
    136-139 Float32     plane 1 radial phase, radians
    140-143 Float32     plane 2 radial amplitude, micrometres peak
    144-147 Float32     plane 2 radial phase, radians
    148-151 UInt32      t20 timestamp, milliseconds since node startup

    total = 4 + 128 + 16 + 4 = 152 bytes

KNOWN DISCREPANCY -- the FDR deck states a 168-byte payload while the COE
working document specifies 152. This module implements the COE document's 152
bytes because that is the only version with a field-by-field layout written
down. The ICS layer doc lists reconciling this with COE as an open item, and
the .NET parser must not be finalised until it is settled. `PAYLOAD_SIZE` is
asserted on every decode so a node sending the other variant fails loudly here
instead of silently producing 16 bytes of garbage features.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from config import DISPLACEMENT_NAMES, FEATURE_NAMES, N_FEATURES

# '<' little-endian, 'f' float32, 'I' uint32
_STRUCT = struct.Struct("<f" + "f" * N_FEATURES + "ffff" + "I")
PAYLOAD_SIZE = _STRUCT.size
assert PAYLOAD_SIZE == 152, f"payload layout drifted: {PAYLOAD_SIZE} != 152"

FDR_CLAIMED_PAYLOAD_SIZE = 168  # unreconciled -- see module docstring


@dataclass(frozen=True)
class Window:
    """One decoded 20-revolution measurement window."""

    rpm: float
    features: np.ndarray       # float32[32], order per config.FEATURE_NAMES
    p1_amp_um: float
    p1_phase_rad: float
    p2_amp_um: float
    p2_phase_rad: float
    t20_ms: int

    def __post_init__(self) -> None:
        if self.features.shape != (N_FEATURES,):
            raise ValueError(
                f"expected {N_FEATURES} features, got {self.features.shape}"
            )

    @property
    def displacements(self) -> np.ndarray:
        return np.array(
            [self.p1_amp_um, self.p1_phase_rad, self.p2_amp_um, self.p2_phase_rad],
            dtype=np.float32,
        )

    def to_dict(self) -> dict:
        d = {"rpm": self.rpm, "t20_ms": self.t20_ms}
        d.update(dict(zip(FEATURE_NAMES, self.features.tolist())))
        d.update(dict(zip(DISPLACEMENT_NAMES, self.displacements.tolist())))
        return d


def encode(window: Window) -> bytes:
    """Serialise one window to the 152-byte wire format."""
    return _STRUCT.pack(
        float(window.rpm),
        *[float(v) for v in window.features],
        float(window.p1_amp_um),
        float(window.p1_phase_rad),
        float(window.p2_amp_um),
        float(window.p2_phase_rad),
        int(window.t20_ms) & 0xFFFFFFFF,
    )


def decode(raw: bytes) -> Window:
    """Parse a 152-byte payload. Raises on any other length."""
    if len(raw) != PAYLOAD_SIZE:
        raise ValueError(
            f"payload is {len(raw)} bytes, expected {PAYLOAD_SIZE}. "
            f"(The FDR deck claims {FDR_CLAIMED_PAYLOAD_SIZE} bytes -- that "
            f"mismatch is unresolved with COE.)"
        )
    fields = _STRUCT.unpack(raw)
    return Window(
        rpm=fields[0],
        features=np.asarray(fields[1 : 1 + N_FEATURES], dtype=np.float32),
        p1_amp_um=fields[1 + N_FEATURES],
        p1_phase_rad=fields[2 + N_FEATURES],
        p2_amp_um=fields[3 + N_FEATURES],
        p2_phase_rad=fields[4 + N_FEATURES],
        t20_ms=fields[5 + N_FEATURES],
    )


def is_valid(window: Window) -> tuple[bool, str]:
    """Host-side sanity gate.

    COE already discards windows with lost DMA data, a missing Keyphasor
    boundary, or out-of-band speed -- a window that arrives here is one the
    node considered valid. This checks only what the host can independently
    verify: that the numbers are finite and the speed is near a supported mode.
    """
    from config import OPERATING_RPM, RPM_TOLERANCE

    if not np.all(np.isfinite(window.features)):
        return False, "non-finite value in feature vector"
    if not np.isfinite(window.displacements).all():
        return False, "non-finite value in displacement fields"
    near_mode = any(
        abs(window.rpm - mode) <= RPM_TOLERANCE * mode for mode in OPERATING_RPM
    )
    if not near_mode:
        modes = " or ".join(f"{m:.0f}" for m in OPERATING_RPM)
        return False, f"RPM {window.rpm:.1f} outside the steady bands ({modes})"
    return True, ""
