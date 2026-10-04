---
paths:
  - "AI-engine/**"
  - "AI-test/**"
---
# Rule: AI engine code

- Every constant traces to a team document: `Project architecture/**` or `Specifications.md`.
  `aiengine/config.py` mirrors those docs. Fix the doc first, then config.
- Model unit = one biaxial bearing (16 raw features: X then Y, each in COE's 8-feature order).
  Rig windows split into bearing 1 = indices 0–15 and bearing 2 = indices 16–31.
- Per-window feature engineering must stay C#-portable: arithmetic on the window plus a stored
  baseline, no fitted sklearn objects in the per-window path. Anything fitted (e.g. PCA, if ever
  used) must be serialised into `model_contract.json`.
- Splits by bearing/run only (`splits.py`). Never window-level. Test bearings are frozen in
  `splits.json` and never used for early stopping or threshold calibration.
- Output-channel → class name always comes from the dataset's label encoder (alphabetical).
- Every report states its data provenance (which dataset, real vs synthetic) on its face.
- Run Python through the shared venv `AI-test/.venv/Scripts/python.exe` (torch CPU-only, installed
  from the CPU index before requirements). Add new deps to `AI-engine/requirements.txt` too.
- Seeds are fixed. Every result file records the seed, git commit, dataset hashes or counts,
  and the command line.
