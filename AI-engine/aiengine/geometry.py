"""Bearing geometries per data source -> characteristic orders (multiples of shaft speed).

Every value is cited. Orders feed dsp.PipelineConfig.bearing_orders so the order-band
features (FTF/BSF/BPFO/BPFI) integrate at each dataset's own defect frequencies.
"""

from __future__ import annotations

import numpy as np

from .config import RIG_BEARING_ORDERS


def orders_from_geometry(n: int, d: float, D: float, contact_deg: float = 0.0) -> dict[str, float]:
    """Standard kinematic bearing orders. d = rolling-element diameter, D = pitch diameter."""
    x = (d / D) * np.cos(np.deg2rad(contact_deg))
    return {
        "FTF": float(0.5 * (1 - x)),
        "BSF": float((D / (2 * d)) * (1 - x**2)),
        "BPFO": float((n / 2) * (1 - x)),
        "BPFI": float((n / 2) * (1 + x)),
    }


BEARING_ORDERS: dict[str, dict[str, float]] = {
    # [TEAM DOC] COE fixed design values (ME-supplied).
    "rig": dict(RIG_BEARING_ORDERS),
    # [CITED] XJTU-SY: LDK UER204, 8 balls, ball dia 7.92 mm, pitch dia 34.55 mm, 0 deg.
    # Wang, Lei, Li, Li, IEEE Trans. Reliability 69(1) 2020, Table I.
    "xjtu_sy": orders_from_geometry(8, 7.92, 34.55, 0.0),
    # [CITED] IMS: Rexnord ZA-2115 double-row, 16 rollers/row, roller 0.331 in, pitch 2.815 in,
    # contact angle 15.17 deg. Qiu et al., J. Sound Vib. 289 (2006) 1066-1090.
    "ims": orders_from_geometry(16, 0.331, 2.815, 15.17),
    # [CITED] MaFaulDa (SpectraQuest MFS): orders as published on the dataset page
    # (www02.smt.ufrj.br/~offshore/mfs/page_01.html), CPM/rpm.
    "mafaulda": {"FTF": 0.3750, "BSF": 1.8710, "BPFO": 2.9980, "BPFI": 5.0020},
}
