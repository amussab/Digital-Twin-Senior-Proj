# COE layer: the sensor node and its laptop scripts

This folder holds everything from the COE side. The AI code lives in `AI-engine/`, and it doesn't depend on anything here.

The **node** (sensors → filter → AD7606 ADC → STM32 → ESP32) measures the two bearings and
sends **one window of 32 features about every 0.72 s** (250 RPM) over its own Wi-Fi.
The two scripts below receive those windows on a laptop:

| Folder | Use it to | Run |
|---|---|---|
| [`training/`](training/README.md) | record **labelled** data (Healthy, Inner race, Outer race, Ball, Cage) and download a CSV | `python training_recorder.py`, then open http://localhost:8000 |
| [`inference/`](inference/README.md) | send each **live** window to your AI model | put the model in `model_hook.py`, then `python inference_receiver.py` |

Run **one of them at a time**: they use the same port and the same node. Both need only Python 3.8 or newer.

Design documents for the node: `Project architecture/COE/`.

## Connect to the node (both scripts)

1. Switch on the 24 V supply. The node starts by itself.
2. Connect the laptop to the Wi-Fi **COE-NODE** (no password). The laptop has no internet on it; that is normal.
3. Start the script. The node finds the laptop by itself, so you don't need an IP address.
   The first time, Windows asks about the firewall. Click **Allow**.

**The ESP board's LED** (it takes about 3–10 s after power-on):
- red, briefly: the ESP is starting.
- blue: COE-NODE is up.
- green: data is arriving from the STM32.

**Problems**
- COE-NODE doesn't appear and the LED is **off**: the ESP's 5 V or GND jumper wire is loose. Push it in firmly or replace it.
- The LED **keeps flashing red**: the 5 V supply dips. Tell COE.
- The LED **stays blue**, never green: press the STM32's black **RESET** button.
- COE-NODE is there but no data arrives: run this once in an **administrator** terminal, then restart the script:
  ```
  netsh advfirewall firewall add rule name="COE node TCP 8000" dir=in action=allow protocol=TCP localport=8000 profile=any
  ```

## What a window contains

- 32 features, always in this order: for each channel `B1X, B1Y, B2X, B2Y`, the features
  `rms, kurtosis, crest, envelope_rms, ftf, bsf, bpfo, bpfi`. Names: `B1X_rms` … `B2Y_bpfi`.
- rms, envelope_rms and the four band values are in **g**; kurtosis and crest are ratios.
- **B2X is a copy of B1Y**: the 4th sensor has no cable, so don't treat B2X as independent.
- The speed is set in the firmware: **250 RPM**, the lab machine's speed. Changing it needs a COE reflash.

`host_receiver.py` (in both folders, same file) is the node's protocol: the node sends an
HTTP POST of JSON to `/features` and waits for `200 {"accepted": true}`.
