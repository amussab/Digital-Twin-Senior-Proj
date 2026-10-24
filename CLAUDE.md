# Digital-Twin-Senior-Proj — repo instructions

Git repo (remote `github.com/amussab/Digital-Twin-Senior-Proj`; **current work branch
`feature/up2-prototype`** for Update Presentation 2, cut from `feature/integration`; never push to
`main`/`feature/dashboard`) for Team M001's senior design build: a
real-time edge digital twin for bearing fault diagnosis and RUL prediction in OH-2 centrifugal
pumps. This is the **engineering repo** — architecture docs and working code. The course
paperwork (syllabus, rubrics, report drafts, deadlines) lives one level up in
`Term261/ICS414/`, which has its own `CLAUDE.md` and is a separate, non-git directory. Read
that one for course/deadline/report context; read this one for the system design and the
`AI-test/` codebase.

## Always-loaded context

@.claude/FACTS.md

- **Specs:** `Project architecture/overview/Specifications.md` is the Markdown copy of the team
  xlsx. Read it before claiming any spec is met.
- **Evidence:** `.claude/rules/evidence-strength.md` holds the course's 0–9 evidence ladder. Every
  PPR artifact targets Rank 9 (a working demo run live at the presentation). The fallback is
  Rank 6–8 (demo or simulation + recorded data). Never ship Rank 0–3 as prototype evidence.
- **Path-scoped rules:** `.claude/rules/ai-engine.md` (AI-engine/**, AI-test/**),
  `.claude/rules/dotnet.md` (src/**), `.claude/rules/git-workflow.md` (always),
  `.claude/rules/up2.md` (UP2 work: provenance tags, hardware-ready boundary, split discipline).
- **Resume ledger:** `.claude/UP2_STATE.md` — read first in every session; it lists done/next steps and
  exact commands. Update it after each step.
- **UP2 deliverables:** `AI-engine/notebooks/UP2_Evidence.ipynb` (+ `.html`), `evidence/UP2_Evidence.md`
  (coach summary), `AI-engine/reports/up2_*` (spec checks), root `README.md` (results, Docker, checklist).
  PPR (earlier) deliverables stay as history.

## Working model: coordinator + specialist subagents

The main session acts as **coordinator**. It owns the plan, the interface contracts
(`AI-engine/DESIGN.md`), integration, and final verification. Specialised work goes to the
subagents defined in `.claude/agents/`:

| Agent | Model | Scope |
|---|---|---|
| data-engineer | sonnet | dataset download, adapters → window table |
| ml-engineer | opus | hybrid N-HiTS + TFT training, evaluation, ONNX export |
| backend-engineer | sonnet | ASP.NET Core + SignalR + ONNX Runtime backend |
| digital-twin-engineer | sonnet | initial physics twin + physics RUL (IS2) |
| evidence-reviewer | sonnet | read-only; grades artifacts against the evidence ladder and PPR rubric |

UP2 added work not covered by those profiles (run as general-purpose agents with the files above as
their reading list): a COE-pipeline engineer (`AI-engine/aiengine/coe/`), a COE-domain-adaptation engineer
(`coe_adapt*`), an evidence author (notebook + `evidence/`), a DevOps/docs engineer (Docker, README).
Ownership is by directory so parallel agents never edit the same file.

How delegation works:
- Each subagent prompt names the files to read first and the exact output contract.
- Subagents report back measured facts only. The coordinator re-verifies key numbers before
  repeating them to the user.
- An evidence-reviewer pass is mandatory before anything goes to the instructor.

## Layout

```
Digital-Twin-Senior-Proj/
├── Project architecture/     # source-of-truth design docs, one folder per discipline
│   ├── overview/              Project Overview.md — cross-discipline summary, read first
│   ├── COE/                   edge acquisition & processing node (STM32/ESP32-C6)
│   ├── ICS/                   AI models + dashboard/digital-twin backend (this user's discipline)
│   └── ME/                    physical system + FE digital twin (placeholder, not yet written)
├── AI-test/                   # HISTORICAL earlier synthetic-only testbench (superseded by AI-engine)
├── AI-engine/                 # hybrid AI engine — see AI-engine/DESIGN.md
│   ├── aiengine/coe/          # UP2: COE edge pipeline as code (sources, 30 kS/s DSP, 152-byte payload, UDP sender, verify)
│   ├── aiengine/synth_v3/     # UP2: SYNTHETIC run-to-failure + seeded-fault generator (coach-approved substitute for rig data)
│   ├── aiengine/up2*.py       # UP2 train/eval/export (v3, v3b deployed GBM heads), up2c* (physics-informed v3c, research),
│   │                          #   up2d* (reworded IS2), coe_adapt*/coe_dataset (COE-domain classifier + cross-bearing gate)
│   ├── models/                # v2 (PPR, public data), v3/v3b (UP2 deployed), v3c (research), v3b_coe (COE-source demo)
│   ├── notebooks/             # PPR_ICS_Evidence (history) and UP2_Evidence.ipynb/.html (live 17-item scorecard)
│   └── .venv/                 # gitignored; torch CPU + xgboost/lightgbm/onnxmltools/pytest/jupyter
├── src/ tests/ tools/         # ASP.NET Core backend + SignalR, Blazor dashboard, C# twin, DemoMeasure
├── docker/ docker-compose.yml # everything runs through Docker (profiles: default, coe, ai, evidence, notebook, test)
├── scripts/                   # run-up2-demo.ps1, run-up2-coe-demo.ps1, resume/setup-auto-resume.ps1
├── evidence/                  # screenshots, integration measurements, UP2_Evidence.md
└── Component selection (pugh matrix)/
```

**`Project architecture/*.md` is the source of truth for specs, targets, and design rationale.**
`AI-test/config.py` mirrors constants out of it (bearing orders, spec targets, feature layout) —
if a `Project architecture` doc changes a number, update `config.py` to match, not the other way
around. These docs get extracted/updated by the user from the team's shared context and FDR
materials; re-read them if the user says they've been updated before trusting your own memory of
their contents.

## AI-engine — current state (UP2, updated 2026-10-25)

`AI-engine/` is the working ML/DSP code; `AI-test/` is the HISTORICAL first testbench (kept for
reference, its "ICS1 fails at ~38%" finding is superseded). Read `.claude/UP2_STATE.md` for the live
ledger and `evidence/UP2_Evidence.md` for the coach-facing summary. Short version:

- **Deployed models (UP2):** v3b = gradient-boosted heads (XGBoost/LightGBM exported to ONNX) in the
  N-HiTS/TFT slots, with a C#-portable post-process (stage median + RUL log-EMA). The team and coach
  approved changing from N-HiTS/TFT for performance; the N-HiTS/TFT results on the same data are kept
  in the evidence for transparency (S7 46.9 %). v3c (physics-informed RUL head) is a research artifact,
  NOT deployed. `v3b_coe` = classifier retrained on COE-pipeline windows + cross-bearing gate, for the
  COE-simulator demo (stage/RUL display on COE-source data is NOT evidence).
- **Data:** synthetic (`aiengine/synth_v3`, coach-approved substitute because sensor parts may not
  arrive) + COE-simulator windows (`coe_dataset`). Public datasets (XJTU-SY, IMS, MaFaulDa) are
  git-ignored; they live in the sibling checkout `Repo/Digital-Twin-Senior-Proj/AI-engine/data` —
  set `AIENGINE_DATA_DIR` to it. PPR (v2) public-data results are history and stay in the evidence.
- **Hardware boundary:** `SampleSource` -> COE pipeline -> 152-byte payload -> UDP 5005 -> backend.
  Hardware day = replace the sender only (`aiengine/coe/README.md`).
- **Run:** use `AI-engine/.venv/Scripts/python.exe` (or Docker, see README). `python -m aiengine.up2_eval`
  re-scores saved predictions + measures latency; `python -m pytest AI-engine/tests`;
  `dotnet test tests/DigitalTwin.Backend.Tests -c Release` (Release: a stale Debug testhost may hold dlls).
- **Evidence discipline:** every number tagged `[SYNTHETIC]`, `[SIMULATION]`, `[MEASURED host]`,
  `[MEASURED on <dataset>]`, `[CALCULATED]`; fresh test sets are registered in `splits.json` BEFORE fitting
  and scored once (test2, test3, test4, synth_coe); the item count (17, BOM excluded by the reviewer) is
  computed from a registry in `AI-engine/notebooks/up2_helpers.py` (`OVERRIDES` dict for other
  departments' evidence). Never mark an item met without an artifact.

## Standing conventions for this repo

- **All UP2 data is synthetic or simulated (coach-approved). Say so, every time, on anything this
  generates.** No fault-seeded or run-to-failure data from the team's own rig exists yet. Every report `evaluate.py` writes stamps
  this on its face — preserve that discipline in anything else you write, following the course's
  evidence-quality standard (sourced-vs-synthesized labeling, no fabricated test/measured results).
- **"Pretrained, fine-tuned" is not literally accurate for the current code.** `pytorch-forecasting`
  supplies the N-HiTS/TFT *architectures* with no pretrained weights; both models here train from
  initialization on synthetic data. `AI-test/README.md`'s "On the word pretrained" section has the
  precise wording to use — check it before writing report/slide language that says "pretrained."
- **Never assume `config.FAULT_CLASSES`' order is the model's output-channel order.**
  `NaNLabelEncoder` sorts labels alphabetically (`ball, cage, healthy, inner_race, outer_race`),
  not the config's declared order. This was a live, invisible-in-the-loss bug once already (see
  `data.class_order()` and its callers). Any new code that maps a model output index to a class
  name must call `data.class_order()` on the trained dataset, not read the config constant.
- **The COE payload is 152 bytes** (`AI-test/payload.py`), per COE's own working doc — the FDR
  deck's "168 bytes" claim is an unreconciled discrepancy, not a typo to silently pick a side on.
  `payload.decode()` asserts exactly 152 and raises on anything else. Don't relax that assertion to
  make a script pass; surface the mismatch instead.
- **Splits are always by run, never by window** (`data.split_runs`) — a bearing's history is
  autocorrelated, so a window-level split reports accuracy the deployed system won't reproduce.
  Any new evaluation code must reuse `data.split_runs`/`data.subset`, not re-split ad hoc.
- **`.gitignore` excludes `.venv/`, `AI-test/data/`, `AI-test/artifacts/`, `lightning_logs/`, and
  the large `rul_predictions.csv` dump.** `AI-test/reports/*.md` and `.json` (the spec-check
  reports) ARE tracked — they're the evidence record of what was run and scored, not scratch
  output. When cleaning up report files, keep at least the latest run per meaningful change, don't
  bulk-delete the whole directory.
- **`export_onnx.py`'s `model_contract.json` is the only thing the .NET side should read** for
  tensor layout, class order, feature constants, and calibration — never hand the C# team a
  hardcoded copy of a Python constant. If you add a new feature-engineering constant, add it to the
  contract in the same change.
- Feature engineering (`features.py`) must stay reimplementable in a few dozen lines of C# — no
  fitted scalers, no pandas in the per-window path. This is a hard constraint, not a style
  preference: it's what lets B9 run both models in-process without a Python sidecar.

## Open items (don't silently resolve these — they need the user or the team)

1. **IS2 wording** — coach said the spec is unclear; reworded v2 text lives in
   `Project architecture/overview/Specifications.md` (v1 text kept, changelog). The official xlsx still
   needs the same edit by the team. Never present the old literal "max <=5 %" as met.
2. **Spec naming N-HiTS/TFT vs deployed GBM heads** — approved by the user/coach for performance;
   keep the comparison visible.
3. **Other departments' evidence** (C1, C2, C6, S1, S2, S3) — to be supplied by owners; add to
   `up2_helpers.OVERRIDES`. BOM (C3) is excluded by the reviewer; item count is 17, bar >=13.
4. **COE stage/RUL head on real COE features** needs refitting on rig commissioning + run-to-failure data;
   `tau` of the cross-bearing gate likewise. Real Wi-Fi hop, STM32 timing and laptop-vs-Pi-5 latency
   are unmeasured (IS1 is design-level until then).
5. Official course report (second instalment) is separate and later; this repo's evidence feeds it.
