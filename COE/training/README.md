# Training recorder (COE node → labelled CSV)

A web page for recording training data. Pick the bearing condition, press Start, run
the machine, press Stop, then download a CSV.

Conditions: **Healthy, Inner race, Outer race, Ball, Cage**.

## Steps

1. Switch on the node and connect the laptop to the Wi-Fi **COE-NODE**
   (details, LED colours and problems: [../README.md](../README.md)).
2. Run:
   ```
   git pull
   cd COE/training
   python training_recorder.py
   ```
3. Open **http://localhost:8000**. The top pill turns green, showing **"Data arriving"**.
4. Choose the condition, add a note if useful, and press **Start**. Press **Stop** when the run is done.
5. Download the CSV for one run or for all runs.

## The data

- Files are saved in `COE/training/training_data/`. Git ignores this folder,
  so copy it out yourself. Each run is saved as its windows arrive, so stopping the
  script loses nothing.
- CSV columns: `run_id, label, note, received_at, rpm, window_id, sequence,
  device_id, valid_channels`, then the 32 features `B1X_rms` … `B2Y_bpfi`
  (order and units: [../README.md](../README.md)).
- The labels are whatever you select. Record one condition per run, and record healthy runs too.
