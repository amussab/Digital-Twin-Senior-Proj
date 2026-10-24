# UP2 final review (2026-10-25, read-only reviewer)

Checked against the PPR deliverables (DT-integration), the ledger, FACTS, rules, evidence file, first review and Specifications.md (IS2 v2).

## Verdict
Package is at or above PPR parity and honest overall. No contradiction in headline numbers. Remaining issues are stale text in rule/ledger files and a few wording items. Nothing blocking.

## 1. Parity
- README: what, architecture (mermaid), model table, results, Docker quick start, native quick start, demo scripts, hardware-day swap, presenter checklist, provenance box, PPR history: all present.
- `docker compose config -q` passes (rc 0). Services: default `backend`; `coe` backend-coe+coe-sim; `ai` ai-engine; `test` test+test-dotnet; `evidence`; `notebook`. Matches README/FACTS. Images were not built here, so "81 tests pass in Docker" is unverified by me; `pytest AI-engine/tests` natively: **81 passed** (126 s), matching the claim.
- `python -m aiengine.up2_eval` runs and prints the table (S7 15.53/46.94 history rows, S8 0.39 ms, IS1 leg 4.2 ms). It writes a new `up2_spec_check_*.md/json`; I deleted my copies, tree is clean.
- Notebook: 29/29 code cells have execution counts and outputs, 0 error outputs. HTML present (2.5 MB). HTML and notebook tally agree: MET 11 of 17 = 64.7 %, conservative 9 of 17 = 52.9 %, 2 items still needed. PPR notebook also 29/29, 0 errors.

## 2. Number consistency (all identical across README, UP2_Evidence.md, notebook, FACTS unless listed)
OK: S7 12.64 / 10.75 / 13.17 live; IS3 F1 0.959 (0.9587), stage max 1, 80/80; N-HiTS 46.9; TFT F1 0.924, stage 3; IS1 p95 ~98; hop limit ~458; IS2 v2 16.3 / 49.2, thresholds 18 / 63, v3c 18.3, v1 233 / 216, exact-AI 18.1, AUROC 0.55; COE 0.584 -> 0.875, 0.6 %, 22.8 %; 11/17, 7 firm + 4 design, bar 13.
Mismatches:
1. S8 canonical: docs say 0.35 ms; notebook live output says 0.334 ms; `up2_eval` just gave 0.392 ms. Run-to-run noise, but "0.35" is not in the saved output.
2. TFT p95: README/Evidence say 1.54 ms (stored value); `up2_eval` live measures 12.14 ms (and PPR TFT 11.1 ms). Differs 8x; the docs' own S8 caveat lists other values but not 12 ms.
3. IS1 p95: 98.1 (notebook) vs 97.1 (v3b integration md) vs 97.0 (COE md). Fine as "~98", just cite the file.
4. Notebook IS2 literal v1 row = 302.1 % (test2, tuned filter) vs README 233 % (test4, v3b). Different sets, but a reader will see two "v1 max" values.
5. Run-time of the notebook: README 3-4 min, Evidence.md ~3 min, FACTS ~6 min.

## 3. Honesty audit
- Strong points: synthetic/simulated statement first, tags on rows, IS2 caveat box visible (README row, Evidence 6c, FACTS), public-data FAIL kept, N-HiTS/TFT vs GBM disclosed, test-set exposure counts stated (old set twice, test2 once + re-score, test3 once, test4 once).
- Claims stronger than evidence: (a) Evidence.md says IS1 "Rank 9 as a simulation" and a live demo, while the first review graded it 6 and the hop is injected; README is more careful. (b) "MET" for S7/IS3 is on synthetic data; the spec text names NHiTS/TFT, which fail (46.9 %, stage 3). Disclosed, but the README results table shows only "original N-HiTS 46.9 % (worse)" and not that the literal spec reading is FAIL. (c) The spec's IS2 v2 is not in the official xlsx; README says so, good. (d) "hardware-ready" is only claimed as "boundary"; OK.
- Stale: `.claude/rules/up2.md` still says "all 18 items (>=14)"; `UP2_STATE.md` goal line says ">=14 of 18" and log lines 45-57 say 10/17 and IS2 NOT MET (log is dated history, but the goal line and step 9 unchecked are stale); `Specifications.md` PPR section says 18 items / 10; FACTS top "Specs" line still lists IS2 <=5 % and S7 "(N-HiTS)" without v2/GBM note. CLAUDE.md is current. No "v3c deployed" claim found; README correctly says v3c not deployed.
- Evidence.md section 5 has a duplicated sentence ("The physics-informed v3c attempt (still FAIL under v1). The physics-informed v3c ...").

## 4. Repo hygiene
- Files > 5 MB tracked: `AI-engine/models/v3_rul.onnx` 6.0 MB, `v3b_rul.onnx` 10.4 MB, `v3c_rul.onnx` 9.7 MB (violates git-workflow rule "< 2 MB"; v3b is the deployed one so it is needed; v3/v3c are history/research), `babylon.js` 8.6 MB (vendored). Notebook .ipynb 2.0 MB, html 2.5 MB: acceptable.
- No secrets matched (regex scan of tracked non-notebook files). No .venv, checkpoints, pycache, or lightning_logs tracked. `.claude/settings.local.json` is untracked (leave out of commits; consider gitignore).

## Fixes, ordered by importance
1. Add one sentence in README results table S7/IS3/S8 rows: the literal spec models (N-HiTS/TFT) fail on the same data; MET refers to the approved GBM heads.
2. Change Evidence.md IS1 wording from "Rank 9" to design level / Rank 6-9 with injected hop, to match README.
3. Fix `.claude/rules/up2.md`, `UP2_STATE.md` goal line, FACTS "Specs" line (17 items, bar 13, IS2 v2, GBM heads); tick or annotate step 9.
4. Reconcile S8/TFT latency: quote the range (0.3-0.4 ms classifier; TFT 1.5-12 ms depending on run/threads) or re-measure once and use one number everywhere.
5. Label the notebook's 302 % IS2 v1 row as test2 / tuned filter and README 233 % as test4 v3b, or drop one.
6. Unify the notebook run-time (3-6 min) across README, Evidence.md, FACTS.
7. Remove the duplicated sentence in Evidence.md section 5.
8. Consider dropping `v3_rul.onnx`/`v3c_rul.onnx` from the tree or documenting the >2 MB exception; add `.claude/settings.local.json` to .gitignore.
