# Fact block — Team M001 engineering repo

Durable facts that must survive context compaction. Each fact is stamped with its source.
Update a fact in place when it changes, with the new date. Never delete one silently: strike it
through and note what replaced it.

## Project
- Title: "Real-Time Edge Digital Twin for Bearing Fault Diagnosis and RUL Prediction in OH-2
  Centrifugal Pumps". KFUPM Senior Design II (ICS414), Team M001, term 261. [course CLAUDE.md]
- ICS models owner: Muhammad Jaddoua (this user). Dashboard/UI owner: Abdulrazaq "Mussab". COE:
  Abdullah Ewidah. ME: Abdelrahman Elshenawy. ICS reviewer: Dr. Hani Al-Mohair. ME reviewer:
  Dr. Fadi Al-Badour (harshest). Coach: Dr. Uthman Baroudi. [course CLAUDE.md]
- Specs: `Project architecture/overview/Specifications.md` (copy of the xlsx, 2026-10-04).
  ICS-owned: S7 RUL MAPE ≤15% (N-HiTS), S8 classification <200 ms/window (TFT), S9 dashboard
  ≥10 Hz, C5 LAN-only dashboard. Integrated: IS1 <500 ms end-to-end, IS2 physics-AI residual ≤5%,
  IS3 macro-F1 ≥85% / stage error ≤1 / caught by stage 3.

## Rig / signal chain (COE docs on main, updated 2026-09-17)
- 2 radial bearings × biaxial (X,Y) IEPE accelerometers = 4 channels. No Z axis.
- AD7606 at 30 kS/s, oversampling disabled. External 4th-order MFB LPF at ~10 kHz.
  Band-pass 2–8 kHz. Envelope at 5 kS/s (decimation ×6). 4096-pt FFT.
- Window = 20 shaft revolutions (1200/RPM s). Steady modes 1750 and 3600 rpm.
- Payload 152 bytes: rpm f32 + 32 features f32 + 4 displacement f32 + t20_ms u32 (FDR deck says
  168, unreconciled).
- Feature order per channel: bp_rms, kurtosis, crest, env_rms, FTF, BSF, BPFO, BPFI band
  magnitudes. Channels: B1-X, B1-Y, B2-X, B2-Y (indices 0-7, 8-15, 16-23, 24-31).
- Rig bearing orders: FTF 0.38, BSF 1.98, BPFO 3.05, BPFI 4.95 (8 elements, d/D ≈ 0.2375).

## AI design decisions
- Hybrid AI = N-HiTS (health-index forecast → RUL by threshold crossing) + TFT (per-window fault
  class). TFT's class selects the RUL failure threshold. [AI-test README]
- Model unit = ONE biaxial bearing (16 raw features). The rig gives two such series per window.
  Decided 2026-10-04. [ICS External Datasets §5]
- Pretraining datasets: XJTU-SY + IMS Test 1 (RUL); MaFaulDa (+ XJTU-SY / IMS late-stage)
  (classification). FEMTO dropped (0.1 s records < 20 rev). CWRU demoted (single axis, 12 kHz).
- `pytorch-forecasting` ships no pretrained weights. "Pretrained" = our own pretraining on public
  data, then fine-tuning.
- Splits by run/bearing, never by window. Class index order comes from the dataset encoder
  (alphabetical), never from the config list.

## PPR (Preliminary Prototype Review), Week 8
- EXPO style. Team M001: Monday (2026-10-05), Room 309, 3:00–6:00 PM. Worth 4%.
  [Weekly Blackboard/Week 7]
- Graded 50% constraints / 50% specs. Per department, Exemplary = more than 1 constraint and 1
  spec met with satisfactory-or-better evidence. Team table, Exemplary = more than 50% of all
  items met.
- Evidence ladder: `.claude/rules/evidence-strength.md`. Aim for Rank 9: a working simulation
  demo, live at the presentation.

## Git
- Remote: github.com/amussab/Digital-Twin-Senior-Proj.
- Branches: main (stable) ← stage ← dev (integration) ← feature/*.
- feature/dashboard belongs to Mussab. **Never commit to, rebase or force-push it.** Read it
  only, or merge it INTO our own integration branch.
- AI work: feature/ai-engine (cut from feature/ai-test + origin/main, 2026-10-04).

## PPR rubric arithmetic (coordinator, 2026-10-04)
- The team table has 6 constraints + 9 specs + 3 integrated = 18 items. Exemplary needs more
  than 50%, i.e. at least 10 items met with satisfactory-or-better evidence.
- ICS Exemplary target: C5 (LAN-only) + C4 (offline AI/processing) = more than 1 constraint, plus
  S8 (TFT latency) and S7 (RUL) as specs. Extra ICS-reachable items toward the 10: S9 (≥10 Hz
  dashboard), IS1 (<500 ms), IS3 (F1/stage/early catch), IS2 (physics-AI residual, simulation
  only).
- Evidence deliverable: `AI-engine/notebooks/PPR_ICS_Evidence.ipynb`. It is a live, executed
  scorecard: every claim is recomputed in a cell from data on disk (Rank 9 when run at the PPR).

## Final PPR state (2026-10-05 ~06:30, branch feature/integration)
- Deliverable: `AI-engine/notebooks/PPR_ICS_Evidence.ipynb` (+ .html fallback). Run-All ~226 s,
  executed at f0aa35a. Root README has the results, the presenter checklist and docker compose.
- MET live (Rank 9): C5 (HTTP + UDP LAN guard), C4 AI/host part (COUNTED for ICS: team
  decision 2026-10-05, no cloud service, AI + dashboard run on the local host; COE-assigned in the spec sheet), S8 TFT p95 11.1 ms, S9 30 Hz (client rate, replay data).
- ICS department = Exemplary: constraints C4 + C5, specs S8 + S9. C3 was rejected for ICS by the reviewer.
- PARTIAL (CONDITIONAL MET): IS1. Backend warms up ONNX at startup (2026-10-05). Host leg, ONNX-active: p50 25.1 / p95 30.0 /
  max 39.8 ms; cold-start one-off 137.5 ms. Budget 333 (COE S5, unmeasured) + 39.8 = 372.8 ms, 127 ms left for the Wi-Fi hop
  (unmeasured). Team decision: counted as MET once COE confirms a measured leg <= 333 ms and the Wi-Fi hop is measured.
- NOT MET: S7 281.8 % (LOBO median 73.7 %); IS3 5-class macro-F1 0.581 (MaFaulDa 0.848,
  XJTU 0.450, IMS 0.679), stage max error 4, 1/4 caught by stage 3; IS2 (simulation residual
  ~100 %).
- Models: v2 = XJTU-SY + IMS + MaFaulDa, export a78a828. C# golden parity passes.
  v2 was the second look at the XJTU/IMS test bearings (disclosed).
