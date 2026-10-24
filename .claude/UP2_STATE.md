# UP2 resume ledger (single source of truth for resuming after an interruption)

Updated by the coordinator after every step. If you are a fresh session: read this, `.claude/FACTS.md`,
`.claude/rules/up2.md`, then do the first unchecked item. Do not redo checked items; verify them with
the listed command if in doubt.

Goal: prototype fully working for UP2 demo; covers old (PPR) met specs + all ICS specs + integrated
specs; >=14 of 18 items MET with evidence; hardware-ready source boundary; UP2 evidence file.

## Steps
- [x] 0. Worktree/branch `feature/up2-prototype` created; main docs pulled; rules/FACTS written
- [x] 1. COE pipeline in code: `AI-engine/aiengine/coe/` (raw 4ch@30kS/s -> BPF -> envelope -> FFT -> 152B payload), SampleSource abstraction, tests, COE1/COE2/COE3/C6 measurements
- [ ] 2. Synthetic run-to-failure + fault data generator through the COE pipeline (provenance-tagged)
- [ ] 3. Models: reach ICS1 (MAPE<=15), ICS3/I3 (macro-F1>=85, stage err<=1, caught by stage 3) with XGBoost/AutoGluon/N-HiTS/TFT; ONNX export + golden parity
- [ ] 4. Physics twin <-> AI residual (I2 <=5%, stage 1->3) on synthetic
- [ ] 5. Backend + dashboard wired to new models and pipeline source (IS1 end-to-end <500 ms measured incl. Wi-Fi hop sim); hardware source stub documented
- [ ] 6. UP2 evidence notebook + `evidence/UP2_Evidence.md` (old vs new met table, provenance, rank)
- [ ] 7. evidence-reviewer pass; fix findings
- [ ] 8. Commit + push `feature/up2-prototype`; demo script `scripts/run-up2-demo.ps1` verified end-to-end
- [ ] 9. Final report to user

## Log (append, newest last)
- 2026-10-24: setup done.
- 2026-10-24: datasets found at sibling checkout (see FACTS); ML agent told to use AIENGINE_DATA_DIR.
- 2026-10-24: step 1 done. AI-engine/aiengine/coe/ (sources, pipeline, udp_sender, run, verify, README), tests/coe (24 pass), reports/coe_verification.{md,json}. Run: `python -m pytest tests/coe -q`, `python -m aiengine.coe.verify` (~25 min, 1-h C6 sim). COE1/COE3 MET (design, simulation); COE2 host max 9.0 ms, STM32 est. 28 ms pessimistic (calculated); C6 0.13% end-to-end (assumed loss rates). Built with system python (venv lacked numpy at the time).
