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
21:42:15.011  window    118  1750.3 RPM (20 turns), PROX1 gap 1227 um, PROX3 gap 1024 um  ->  healthy   (model 3 ms; end-to-end 62 ms OK (limit 500: STM32 4 + UART 19 + ESP32 0 + Wi-Fi 35 + laptop 4))
```
**End-to-end latency** (spec: under 500 ms from the window's last sample to the AI result; collecting
the window is not counted). The node (firmware schema 3, 16 Nov 2026) sends its part in the JSON:
STM32 processing, UART wire time, ESP32 queue time and the Wi-Fi time (the previous request's round
trip, a close estimate). The receiver adds the laptop's time (POST received -> `predict()` done)
and prints the total per window; Ctrl+C prints the maximum and how many were over 500 ms.
`window` also has `window['measured_rpm']` (the Keyphasor speed, or None), `window['rpm']` (the
speed the features were computed for, whole RPM) and `window['shaft']` (the probe values by name,
e.g. `window['shaft']['prox1_gap_um']`). These are for the digital twin; the model needs only
the 32 features.

## For the digital twin: GET /digital-twin

While the receiver runs, **http://localhost:8000/digital-twin** (from another computer: the receiver
laptop's IP instead of `localhost`) returns the newest window's machine speed and proximity probe
readings as JSON. There is one new reading per window (every 0.69 s at 1750 RPM, 0.33 s at 3600 RPM),
so ask about once per second. Any program or web page can read it (CORS is open).

```json
{"available": true, "age_s": 0.4, "window_id": 1356, "received_at": "2026-11-16T01:58:12.411+03:00",
 "rpm": 1747, "measured_rpm": 1749.3, "revolutions": 20, "shaft_flags": 31, "sample_rate_hz": 29998.5,
 "prox1": {"gap_um": 1225.9, "pkpk_um": 24.9, "x1_um_pp": 16.4, "x1_phase_deg": 63.5, "valid": true, "x1_valid": true},
 "prox3": {"gap_um": 1024.0, "pkpk_um": 26.8, "x1_um_pp": 19.3, "x1_phase_deg": 70.3, "valid": true, "x1_valid": true},
 "node_latency": {"stm_ms": 4.2, "uart_ms": 19.271, "esp_ms": 0.15, "wifi_ms": 35.0, "node_ms": 58.274},
 "ai_result": {"window_id": 1356, "result": "healthy", "host_latency_ms": 3.8, "end_to_end_ms": 62.1,
               "latency_limit_ms": 500.0, "within_limit": true}}
```

- `measured_rpm`: machine speed measured by the node from the Keyphasor (ADC channel 7); `null` if
  the node has no steady speed. `rpm` is the whole-RPM speed the features were computed for.
- `revolutions`: whole shaft turns in the window. `sample_rate_hz`: ADC samples per second per
  channel (29998.5; `null` with older firmware).
- `prox1` (ADC channel 5) and `prox3` (channel 6): each probe measures the **distance between its tip
  and the shaft**, in µm. `gap_um` = average distance (where the shaft sits), `pkpk_um` = how much it
  swings while turning (vibration, peak to peak), `x1_um_pp` = the part that repeats once per turn,
  `x1_phase_deg` = where in the turn the shaft comes closest to the probe, measured from the
  Keyphasor mark (0-360°). Use a value only when its `valid` / `x1_valid` is true.
- `age_s`: seconds since this window arrived. If it keeps growing, no data is coming in.
- `ai_result`: the model's answer for the latest window it finished (text), with the end-to-end
  latency of that window (`end_to_end_ms` = the node's part + `host_latency_ms`; `within_limit` =
  under 500 ms). `node_latency` (ms per part) and the latency values are `null` with older firmware.
- Before the first window: `{"available": false, "ai_result": null}`.

## Good to know

- If `predict()` raises an error, the error is printed on that line and the next window still arrives.
- The model runs apart from the receiving part, so a slow model never delays the node. If it falls
  more than 50 windows behind (about 34 s at 1750 RPM), the oldest are dropped and a message says how many.
- If no data arrives, the script prints a message after 5 s.
- Don't run the training recorder at the same time.
