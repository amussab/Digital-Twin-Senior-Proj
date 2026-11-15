# Inference receiver (live COE node → your AI model)

Receives each window from the node **live** (every 0.69 s at 1750 RPM, 0.33 s at 3600 RPM) and passes it to your
model. It prints one line per window with the model's answer.

## Steps

1. **Put your model in `model_hook.py`.** It is the only file to edit:
   - `setup()`: load the model once and return it.
   - `predict(window, model)`: return the result for one window. `window['features']` is
     the 32 values in the fixed order. `window['named']['B1X_rms']` gives one value by name.
   Until you add a model, it prints a short summary instead, so you can test the link first.
2. Switch on the node and connect the laptop to the Wi-Fi **COE-NODE**
   (details, LED colours and problems: [../README.md](../README.md)).
3. Run:
   ```
   git pull
   cd COE/inference
   python inference_receiver.py
   ```
   There is no speed to set: the node measures it from the Keyphasor (ADC channel 7).
   Use the Python that has your model's packages installed.

You will see one line per window:
```
21:42:15.011  window    118  1750.3 RPM, PROX1 gap 1227 um, PROX3 gap 1024 um  ->  healthy   (3 ms)
```
`window` also has `window['measured_rpm']` (the Keyphasor speed, or None), `window['rpm']` (the
speed the features were computed for, whole RPM) and `window['shaft']` (the probe values by name,
e.g. `window['shaft']['prox1_gap_um']`). These are for the digital twin; the model needs only
the 32 features.

## Good to know

- If `predict()` raises an error, the error is printed on that line and the next window still arrives.
- The model runs apart from the receiving part, so a slow model never delays the node. If it falls
  more than 50 windows behind (about 34 s at 1750 RPM), the oldest are dropped and a message says how many.
- If no data arrives, the script prints a message after 5 s.
- Don't run the training recorder at the same time.
