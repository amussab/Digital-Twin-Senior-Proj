# Training recorder (COE node → labelled CSV)

A web page for recording training data from the COE node. Pick the bearing
condition, press Start, run the machine, press Stop, then download a CSV.
It needs only Python (3.8 or newer), so there is nothing to install.

Conditions: **Healthy, Inner race, Outer race, Ball, Cage**.

## Steps

1. Power the node (STM32 + ADC + ESP32). It starts by itself.
2. On the laptop, connect to the Wi-Fi **COE-NODE**. It has no password.
3. Run:
   ```
   git pull
   cd tools/training_recorder
   python training_recorder.py
   ```
   The first time, Windows asks about the firewall. Click **Allow**.
4. Open **http://localhost:8000**. The top pill turns green, showing **"Data arriving"**.
5. Choose the condition, add a note if useful, and press **Start**. Press **Stop** when the run is done.
6. Download the CSV for one run or for all runs.

While the laptop is on COE-NODE it has no internet. That is normal.

**The ESP board's LED** (it takes about 3–10 s after power-on):
- red, briefly: the ESP is starting.
- blue: COE-NODE is up.
- green: data is arriving from the STM32.

**COE-NODE does not appear?**
- LED **off**: the ESP's 5 V or GND jumper wire is loose. Push it in firmly or replace it.
- LED **keeps flashing red**: the 5 V supply dips. Tell COE.
- LED **stays blue**, never green: press the STM32's black **RESET** button.

**COE-NODE is there but no data arrives?** Run this once in an **administrator** terminal, then restart the script:
```
netsh advfirewall firewall add rule name="COE recorder TCP 8000" dir=in action=allow protocol=TCP localport=8000 profile=any
```

## The data

- Files are saved in `tools/training_recorder/training_data/`. Git ignores this folder,
  so copy it out yourself. Each run is saved as its windows arrive, so stopping the
  script loses nothing.
- CSV columns: `run_id, label, note, received_at, rpm, window_id, sequence,
  device_id, valid_channels`, then 32 features named `<channel>_<feature>`.
  - Channels: `B1X, B1Y, B2X, B2Y`.
  - Features, in order: `rms, kurtosis, crest, envelope_rms, ftf, bsf, bpfo, bpfi`.
  - Acceleration values are in g.
- **B2X is a copy of B1Y**: the 4th sensor has no cable, so don't treat B2X as independent.
- The speed is set in the firmware: **250 RPM**, the lab machine's speed. Changing it needs a COE reflash.
- The labels are whatever you select. Record one condition per run, and record healthy runs too.

`host_receiver.py` is the node's HTTP protocol, and the recorder uses it unchanged.
