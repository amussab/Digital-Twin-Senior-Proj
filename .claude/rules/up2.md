# Rule: Update Presentation 2 (UP2) work

Applies to everything on branch `feature/up2-prototype`. Read `.claude/UP2_STATE.md` FIRST in every
session/agent: it is the resume ledger (what is done, what is next, exact commands).

- UP2 = in-class with coach only, 25/26 Oct 2026. Rubric (Week 9): 50 pts = department's 1 constraint
  + 2 specs met with sufficient evidence; 50 pts = MORE THAN 75% of all 18 items (>=14) met with
  satisfactory-or-better evidence. Evidence ladder: `evidence-strength.md` (target Rank 9 = runs live).
- Hardware (AD7606 + IEPE accelerometers + STM32/ESP32-C6) is NOT ready. Every data path goes through
  the `SampleSource` / payload boundary so a real node replaces the simulator with ZERO downstream
  change: raw 4-ch samples -> COE pipeline -> 152-byte payload -> UDP 5005 -> backend.
- Synthetic data is allowed and expected for UP2, but EVERY number carries a provenance tag
  `[SYNTHETIC]`, `[MEASURED on <public dataset>]`, `[CALCULATED]` or `[CITED]`. Never present a
  synthetic score as measured on the team rig. The previous public-data results (XJTU/IMS/MaFaulDa)
  stay in the evidence file next to the new synthetic ones, not replaced.
- Splits by run, never by window. Test runs are never used for training or model selection.
- Do not edit `feature/dashboard`, `main`, or push to anything except `feature/up2-prototype`.
- Keep C# feature engineering reimplementable (no fitted scalers/pandas in the per-window path);
  `model_contract.json` is the only contract the backend reads.
- Commit small and often on `feature/up2-prototype`; update `.claude/UP2_STATE.md` after each step.
