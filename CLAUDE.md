# Digital-Twin-Senior-Proj — repo instructions

Git repo (branch `main`, no remote configured yet) for Team M001's senior design build: a
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
  `.claude/rules/dotnet.md` (src/**), `.claude/rules/git-workflow.md` (always).

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
├── AI-test/                   # earlier synthetic-only testbench for ICS1 (N-HiTS/RUL) and ICS2 (TFT/classification)
├── AI-engine/                 # production hybrid AI engine (real public data) — see AI-engine/DESIGN.md
└── Component selection (pugh matrix)/
```

**`Project architecture/*.md` is the source of truth for specs, targets, and design rationale.**
`AI-test/config.py` mirrors constants out of it (bearing orders, spec targets, feature layout) —
if a `Project architecture` doc changes a number, update `config.py` to match, not the other way
around. These docs get extracted/updated by the user from the team's shared context and FDR
materials; re-read them if the user says they've been updated before trusting your own memory of
their contents.

## AI-test/ — what it is, current state

A Python testbench that trains, evaluates, and ONNX-exports the two ICS-owned models (N-HiTS for
RUL, TFT for fault classification), built for the production **ASP.NET Core / SignalR** backend
(B9 in the ICS architecture doc) — Python here is the training/verification rig, not the deployed
service. Full detail is in `AI-test/README.md`; don't duplicate that file's content here, read it.

**Environment**: `AI-test/.venv/` (gitignored). PyTorch is CPU-only and must be installed from the
CPU index *before* `pip install -r requirements.txt`, or pip pulls a multi-GB CUDA build:
```
cd AI-test
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r requirements.txt
```
Versions last verified working together: `pytorch-forecasting` 1.8.0, `lightning` 2.6.6, `torch`
2.14.0+cpu, `onnxruntime` 1.30.0. Run everything through `.venv/Scripts/python.exe`, not a bare
`python`.

**Commands**: `app.py selftest` (no model/download needed, ~5s), `app.py all` (generate+train+
evaluate+export, ~5 min CPU), or the individual `generate` / `train` / `evaluate` / `export` /
`replay` / `datasets` subcommands. `--help` on any of them.

**Current spec-check result** (`AI-test/reports/latest.md` has the live numbers — re-run
`evaluate` rather than trusting a stale copy here): ICS2/I1/I3 all PASS with large margin; **ICS1
(RUL, target ≤15% MAPE) FAILS at ~38%**, and the harness's own information-floor calculation shows
~32% is the best any forecaster can do while RUL is expressed in absolute hours (life-length
varies 2.5x across runs, so `RUL = total_life × (1−fraction_consumed)` puts a hard floor on
relative error). Percent-of-life-remaining framing gets it to ~16 points. Three resolution options
are written up in `AI-test/README.md`'s "Why ICS1 fails" section — this needs a team decision, not
a quiet reformulation. Don't re-derive this from scratch; read that section first.

## Standing conventions for this repo

- **All data is synthetic. Say so, every time, on anything this generates.** No fault-seeded or
  run-to-failure data from the team's own rig exists yet. Every report `evaluate.py` writes stamps
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

1. ICS1's target framing (absolute-hours MAPE vs. percent-of-life-remaining vs. reformulating as
   direct supervised regression) — see above.
2. COE payload size: 152 vs. 168 bytes.
3. The ICS layer doc's §2 "physics as AI input vs. independent physics RUL compared via residual"
   contradiction — decides what I2 means and what B9's dashboard actually displays. `AI-test`
   currently takes no position on it.
4. Deployment hardware (laptop vs. Raspberry Pi 5) — latency numbers in any report are from a dev
   laptop and must be re-measured on whichever is chosen before being quoted anywhere official.
5. Channel-count gap for external datasets. **Proposed resolution (2026-10-04)**: the rig is two
   biaxial (X,Y) bearings, and XJTU-SY, IMS Test 1 and MaFaulDa all have two orthogonal radial axes
   per bearing. The plan is to make the per-bearing 16-feature record the model unit, not the
   4-channel rig. See `ICS External Datasets & Literature Precedent.md` §5 for the dataset
   selection. `external.py` has not been updated yet: it still has FEMTO and refuses mismatched
   channels, and `--mirror-plane` stays smoke-test only.
