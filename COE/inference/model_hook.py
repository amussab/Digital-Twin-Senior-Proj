"""PUT YOUR AI MODEL HERE. This is the only file you need to edit.

inference_receiver.py calls:
  setup()          once, at start: load your model here and return it.
  predict(window, model)  once per window from the COE node, live
                   (about every 0.72 s at 250 RPM). Return the result.

`window` is a dict:
  window['features']   list of 32 floats, always in FEATURE_NAMES order
  window['named']      the same 32 values by name, e.g. window['named']['B1X_rms']
  window['rpm']        shaft speed set in the node (int)
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


def setup():
    """Load your model once. Example:
        import joblib
        return joblib.load('my_model.joblib')
    """
    return None


def predict(window, model):
    """Return your model's output for one window. Example:
        return model.predict([window['features']])[0]
    Until a model is added, this returns a short summary of the window.
    """
    named = window['named']
    return (f"no model yet | B1X rms {named['B1X_rms'] * 1000:.2f} mg, "
            f"B2Y rms {named['B2Y_rms'] * 1000:.2f} mg")
