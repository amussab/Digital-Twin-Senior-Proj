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

## Shaft speed

The page has a **Shaft speed** box: type an RPM or press 1800 / 3600, then **Set**. It sends the
command over the STM32 USB cable (must be plugged into this laptop) and shows the node's reply and the
RPM of the latest window. The node has no tachometer, so the value must match the VFD. After a node
reset it is back at 1800. Start is disabled while a change is pending, and the speed cannot be changed
while recording. Same from a terminal: `python node_rpm.py 3600` or `python node_rpm.py --query`
(close any serial terminal first: only one program can use the port).

## The data

- Files are saved in `COE/training/training_data/`. Git ignores this folder,
  so copy it out yourself. Each run is saved as its windows arrive, so stopping the
  script loses nothing.
- CSV columns: `run_id, label, note, received_at, rpm, window_id, sequence,
  device_id, valid_channels`, then the 32 features `B1X_rms` … `B2Y_bpfi`
  (order and units: [../README.md](../README.md)).
- The labels are whatever you select. Record one condition per run, and record healthy runs too.
