# Training recorder (COE node → labelled CSV)

A web page for recording training data. Pick the bearing condition, press Start, run
the machine, press Stop, then download a CSV.

Conditions: **Healthy, Inner race, Outer race, Ball, Cage**. Each run also records the **health level** of the mounted
bearing (0 healthy, 1 incipient, 2 moderate, 3 severe), a free **Bearing ID** (e.g. B-03), and **where the faulty part is**
(both bearings / bearing 1 / bearing 2: training labels the other bearing healthy). Tick **Commissioning baseline** for the
healthy reference run (about 2 min settle + 1 min) at each speed. The **Coverage** table on the page shows minutes per
condition, level and speed.

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
4. **Machine speed and shaft:** nothing to choose. Set the machine to the speed you want; the node
   measures it from the Keyphasor and the page shows it (for example **"Speed (Keyphasor):
   1750.2 RPM"**), with the two proximity probes' gap, pk-pk and 1X vibration.
5. Choose the condition, add a note if useful, and press **Start**. Press **Stop** when the run is done.
6. Download the CSV for one run or for all runs.

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
  (order and units: [../README.md](../README.md)), then the shaft columns
  `schema_version, measured_rpm, revolutions, shaft_flags, prox1_gap_um, prox1_pkpk_um,
  prox1_x1_um_pp, prox1_x1_phase_deg, prox3_...` (empty for windows from old firmware), then (appended at the end) `severity, bearing_id, position, is_baseline`.
  Older CSVs without the last four are still read (by header name).
- The labels are whatever you select. Record one condition per run, and record healthy runs too.
- The `rpm` column is the speed the node used for that window; with the Keyphasor firmware it is measured (whole RPM) and
  `measured_rpm` is the exact value. Record each condition at both machine speeds (1750/1800 and 3600) if the model must work at both.

## Train on the recorded data (one command)

From `AI-engine/` (the `.venv` Python), pointing at this folder or a downloaded CSV:
```
python -m aiengine.data_coverage ../COE/training/training_data        # what is still missing
python -m aiengine.train_real --data ../COE/training/training_data    # train + evaluate + export v4_real
```
Protocol it checks (per speed, 1800 and 3600): 1 baseline run (3 min); 3 healthy runs; each fault at level 2 (minimum) or
levels 1-3 (full): 2 runs on different mounts (different Bearing ID). Record at least 4 min per run: the first 2 min are
dropped as settle, 2 min are used. Use a NEW Bearing ID for every physical bearing; the same bearing is never split between
training and held-out data. Then use it live with `NODE_CONTRACT=v4_real` and `NODE_BASELINE=AI-engine/models/baseline_v4_real.json`.
