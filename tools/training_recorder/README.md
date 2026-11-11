# Training recorder (COE node → labelled CSV)

A small web page for recording training data from the COE node. Pick the
bearing condition, press Start, run the machine, press Stop, then download a CSV.
It uses only the Python standard library (Python 3.8 or newer), so there is nothing to install.

Conditions: **Healthy, Inner race, Outer race, Ball, Cage**.

## What you need

- A laptop with Python.
- A **2.4 GHz** Wi-Fi hotspot named **`COE-NODE`**. A phone hotspot or the Windows
  Mobile hotspot both work. Ask Abdullah (COE) for the password, because it is not kept in this public repo.
- The node (STM32 + ADC + ESP32) powered from its supply. It starts by itself, so
  no one needs to press a button. You don't need to know any IP address: the recorder announces itself on the network,
  and the node finds it within a few seconds.

## Steps

1. Turn on the `COE-NODE` hotspot and connect the laptop to it (skip this if the laptop is the hotspot).
2. Power the node. On the STM32 the green LED blinks fast while it is running.
   A red LED means an error, so power the node off and on again.
3. On the laptop:
   ```
   git pull
   cd tools/training_recorder
   python training_recorder.py
   ```
4. The first time, Windows asks about the firewall: allow it. If no data
   arrives, run this once in an **administrator** terminal:
   ```
   netsh advfirewall firewall add rule name="COE recorder TCP 8000" dir=in action=allow protocol=TCP localport=8000 profile=any
   ```
5. Open **http://localhost:8000**. Wait for the green **"Data arriving"** pill
   (one window about every 0.7 s at 250 RPM).
6. Choose the condition, type a note if useful (load, which bearing, and so on), then
   press **Start**. Press **Stop** when the run is done.
7. Download the CSV for one run or for all runs. Runs are saved as soon as each
   window arrives, so closing the script (Ctrl+C) loses nothing.

## The data

- Files are saved in `tools/training_recorder/training_data/` (git ignores this folder,
  so copy it out yourself).
- CSV columns: `run_id, label, note, received_at, rpm, window_id, sequence,
  device_id, valid_channels`, then 32 features named `<channel>_<feature>`.
  - Channels: `B1X, B1Y, B2X, B2Y`.
  - Features, in order: `rms, kurtosis, crest, envelope_rms, ftf, bsf, bpfo, bpfi`.
  - Acceleration values are in g.
- **B2X is a copy of B1Y**: the 4th sensor has no cable, so don't treat B2X as independent.
- The speed is set in the firmware: **250 RPM**, the lab machine's speed. A different speed
  needs a firmware change and reflash by COE.
- The labels are whatever you select. Record one condition per run, and record healthy runs too.

`host_receiver.py` is the node's HTTP protocol (validation, duplicate handling), and the recorder uses it unchanged.
