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
- Run Python through `AI-engine/.venv/Scripts/python.exe` (UP2 venv: torch CPU + xgboost/lightgbm/onnxmltools/pytest/jupyter; the older `AI-test/.venv` is historical) or through Docker (torch CPU-only, installed
  from the CPU index before requirements). Add new deps to `AI-engine/requirements.txt` too.
- Seeds are fixed. Every result file records the seed, git commit, dataset hashes or counts,
  and the command line.
- UP2 additions: every metric is computed by a pure function over a prediction frame (`up2_metrics.py`)
  so notebook, CLI and tests agree; fresh test sets are registered in `splits.json` BEFORE any fit that
  could see them and scored once; models selected on validation only; deployed contract = v3b
  (`model_contract_v3b.json`), COE-source demo = `model_contract_v3b_coe.json`; the Python reference
  (`engine.py`) and the C# backend are pinned together by golden vectors in `AI-engine/models/`.
