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
   python inference_receiver.py --rpm 1750
   ```
   `--rpm` is the machine speed, **1750** (default) or **3600**. Set the machine's speed controller to
   the same value. The node switches within a few seconds; until then each line ends with
   `[node still switching to ... RPM]`.
   Use the Python that has your model's packages installed.

You will see one line per window:
```
21:42:15.011  window    118  1750 RPM  ->  healthy   (3 ms)
```

## Good to know

- If `predict()` raises an error, the error is printed on that line and the next window still arrives.
- The model runs apart from the receiving part, so a slow model never delays the node. If it falls
  more than 50 windows behind (about 34 s at 1750 RPM), the oldest are dropped and a message says how many.
- If no data arrives, the script prints a message after 5 s.
- Don't run the training recorder at the same time.
