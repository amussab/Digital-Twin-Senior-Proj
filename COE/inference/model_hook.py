"""PUT YOUR AI MODEL HERE. This is the only file you need to edit.

inference_receiver.py calls:
  setup()          once, at start: load your model here and return it.
  predict(window, model)  once per window from the COE node, live
                   (every 0.69 s at 1750 RPM, 0.33 s at 3600 RPM). Return the result.

`window` is a dict:
  window['features']   list of 32 floats, always in FEATURE_NAMES order
  window['named']      the same 32 values by name, e.g. window['named']['B1X_rms']
  window['rpm']        machine speed the node used for this window: 1750 or 3600 (int)
  window['window_id']  window number from the node (int)
  window['valid_channels']  bit mask, 15 = all 4 channels valid
  window['received_at']     time on this laptop (ISO text)

Units: rms, envelope_rms and the four band values (ftf, bsf, bpfo, bpfi) are in g;
kurtosis and crest are ratios. B2X is a copy of B1Y (the 4th sensor has no cable).

Whatever predict() returns is printed next to the window. If predict() raises
an error, the error is printed and the next window still arrives.
"""

CHANNELS = ('B1X', 'B1Y', 'B2X', 'B2Y')
FEATURES = ('rms', 'kurtosis', 'crest', 'envelope_rms', 'ftf', 'bsf', 'bpfo', 'bpfi')
FEATURE_NAMES = [f'{channel}_{feature}' for channel in CHANNELS for feature in FEATURES]


import os
import sys
from pathlib import Path

_AI = Path(__file__).resolve().parents[2] / 'AI-engine'
_live = None


def setup():
    """Load the AI-engine NodeEngine (aiengine.node_live). Env: NODE_CONTRACT (default v3e_node; v4_real after
    training), NODE_BASELINE (baseline JSON, loaded if present and saved when captured), NODE_SETTLE_S, NODE_BASELINE_S.
    Falls back to a no-model summary if the AI-engine cannot be imported."""
    global _live
    if str(_AI) not in sys.path:
        sys.path.insert(0, str(_AI))
    try:
        from aiengine import node_live
    except Exception as error:  # keep the receiver alive; say so loudly
        print(f'WARNING: AI-engine not importable ({type(error).__name__}: {error}); running with NO model.', flush=True)
        return None
    _live = node_live
    model = node_live.setup()
    print(f"NodeEngine ready: contract={os.environ.get('NODE_CONTRACT', 'v3e_node')}", flush=True)
    return model


def predict_full(window, model):
    """Return (engine_output_dict_or_None, one_line_string). The receiver forwards the dict to the backend."""
    if model is None or _live is None:
        named = window['named']
        return None, f"no model | B1X rms {named['B1X_rms'] * 1000:.2f} mg, B2Y rms {named['B2Y_rms'] * 1000:.2f} mg"
    return _live.predict_full(window, model)


def predict(window, model):
    return predict_full(window, model)[1]
