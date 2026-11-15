"""v7_node: learnable-profile S7 research + build. [SIMULATION; endpoints resampled from rig windows, 1 bearing/fault,
curve shape + life variance assumed]. Shared helpers (research and build); heads/export/test live further below / in v7_build.

Everything here reuses v5_node (engineering, heads, post-process) and up2_metrics (S7 definition, unchanged).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import up2
from . import up2_window as uw
from . import v5_node as V5
from .config import ROOT
from .rig_anchor import constants as K
from .rig_anchor import constants_profiles as CP
from .rig_anchor import generator as G
from .rig_anchor import pools as P

COLS, R, NL = V5.COLS, V5.R, V5.NL
CACHE = ROOT / "scratch_v7"


class RFeat:
    """Lite FeatSet (RUL/z features only, NL=48 with HI levels); duck-typed for v5_node head helpers."""

    def __init__(self, tab: pd.DataFrame, X=None, idx=None):
        self.tab = tab
        if X is None:
            X, idx = up2.unit_features(tab, NL, False, columns=COLS, warm=V5.T_CLS, levels=True)
        self.Xr = X
        self.T = up2.merge_truth(idx, tab, [c for c in V5.TRUTH if c not in ("unit_id", "window_index")])
        self.F = uw.n_features(R, False)

    def xr(self, levels: bool) -> np.ndarray:
        return self.Xr if levels else self.Xr[:, :self.F]

    def sub(self, mask: np.ndarray) -> "RFeat":
        o = object.__new__(RFeat)
        o.tab, o.Xr, o.T, o.F = self.tab, self.Xr[mask], self.T[mask].reset_index(drop=True), self.F
        return o


def build_df(profile: str | None, splits=("train", "val"), plan=None, pools=None) -> pd.DataFrame:
    plan = plan or [p for p in G.plan_units() if p["split"] in splits]
    return G.build(pools or P.Pools(), plan, profile=CP.get(profile))


def make_sets(profile: str | None, splits=("train", "val"), plan=None, pools=None) -> dict:
    t0 = time.perf_counter()
    df = build_df(profile, splits, plan, pools)
    tab = V5.engineer(df)
    out = {s: RFeat(tab[tab["split"] == s]) for s in splits}
    print(f"[{profile}] built {sorted(tab.rig_unit.nunique() for _ in [0])} units, {time.perf_counter() - t0:.0f}s", flush=True)
    return out
