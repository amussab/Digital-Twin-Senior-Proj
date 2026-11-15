"""CLI: python -m aiengine.rig_anchor.build --out data/rig_anchor/rig_anchor.parquet  (also registers the split if absent)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import ROOT
from . import constants as K
from . import generator as G
from . import registry


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "data" / "rig_anchor" / "rig_anchor.parquet"))
    ap.add_argument("--wrong-shape", action="store_true", help="control set: concave damage curves (a=0.9, p=1.0..1.2) instead of the assumed family")
    a = ap.parse_args(argv)
    print("split section:", registry.register(), "verified:", registry.verify())
    over = {"a": (0.85, 0.95), "p": (1.0, 1.2)} if a.wrong_shape else None
    df = G.build(shape_override=over)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    meta = dict(df.attrs["provenance"], rows=len(df), wrong_shape_control=bool(a.wrong_shape))
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(K.PROVENANCE, "->", out, "rows", len(df), "units", df["rig_unit"].nunique())


if __name__ == "__main__":
    main()
