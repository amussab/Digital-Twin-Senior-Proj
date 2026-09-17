# Real-Time Edge AI Bearing Monitoring System
## ICS — AI, Fault Classification & Dashboard Layer

Extracted 2026-09-14 from the team's full project context (`Claude outputs/PROJECT_CONTEXT_EXPORT.md`)
and the FDR presentation (`Shared Folder LOCAL/TEAM M001/3-Presentations/3 - FDR Presentation/`).
This file covers only the ICS-owned portion of the system — what the local AI/dashboard host
receives from COE's edge node, what it does with it, and why each choice was made. See the COE
folder's two documents for the edge-acquisition side this layer depends on, and the ME folder for
the physics/FE side referenced in §2 below.

---

### 1. Purpose and Scope

ICS owns the **local AI + digital-twin host**: it receives the fixed payload COE's edge node
produces every 20-revolution window, runs two fine-tuned forecasting models on it, and serves a
live web dashboard. Three ICS specifications and a share of three cross-discipline integrated
specs are owned here (see §4).

- **RUL regression** — pretrained **N-HiTS**, fine-tuned, continuous background prediction.
- **Fault classification** — pretrained **TFT** (Temporal Fusion Transformer), fine-tuned,
  per-window prediction.
- **Local web dashboard + digital twin backend** — ASP.NET Core / SignalR, combines both model
  outputs with the ME-owned physics RUL estimate into one live view.

Team: **Muhammad Salah Jaddoua** (this user, 202246240) owns the two models — RUL regression and
fault classification. **Abdulrazaq Musaab Alsayed Ahmad** ("Mussab", 202267720) owns the dashboard
and digital-twin presentation layer. Both are ICS; reviewer is Dr. Hani Kaid Al-Mohair
(hanik@kfupm.edu.sa).

---

### 2. ⚠️⚠️ OPEN CONTRADICTION — resolve before writing or building anything that touches the physics input

**This affects this layer directly**: it decides what N-HiTS and TFT actually receive as input,
and what the dashboard's "residual"/twin panel is supposed to show.

**Report version** (`report_draft_v1.docx`, corrected 2026-09-03): the physics model (bearing
kinematics/BCFs + Miner's cumulative-damage rule) is **software context feeding N-HiTS's and
TFT's input** — physics-derived features engineered directly into the models' inputs. There is
**no independent physics RUL number**; nothing is compared to the AI's output. The integrated spec
(then called N2, now I2) was defined as an **ablation study**: N-HiTS trained with physics-derived
input features must beat an ablated DSP-only-feature baseline by ≥15% relative MAPE. Under this
framing, TFT's attention-based confidence signal (flagging low-confidence/out-of-distribution
calls) is the safety mechanism referenced in Ch.2.3's ethics section — there is no residual check
to fall back on.

**FDR presentation version** (slides 3, 4, 5, 13, 16, 17 — internally consistent with each other):
the physics model is an **independent, parallel RUL estimator**, built and owned by ME (block B3,
"Stiffness RUL Estimator... K(t) trend, independent of AI"). Its output feeds a **residual check**
against this layer's N-HiTS RUL output. The final spec table's integrated spec **I2** reads
**"Physics-AI RUL residual ≤5%, stages 1→3."** B9's own mechanism description (this layer's .NET
backend) says it "combines the AI outputs with B3's physics RUL and [the] residual into one
payload" — i.e. the dashboard is meant to display and check a residual, not just serve
physics-informed model outputs.

**What this means concretely for this layer, once resolved**:
- If the **AI-input framing** wins: N-HiTS/TFT's feature pipeline needs the physics-derived
  features (Miner's-D, BCFs) as literal model inputs, sourced from COE's payload and/or computed
  on this host; the dashboard has no separate "physics RUL" panel to reconcile against; I2 becomes
  an ablation-study number this layer is directly responsible for producing (train-with vs.
  train-without comparison).
- If the **residual-check framing** wins (what FDR actually showed): this layer's .NET host needs
  to *receive* ME's independently-computed physics RUL (not just physics features) as an
  additional input to B9, compute the residual against N-HiTS's own RUL output, and render both
  numbers plus the residual on the dashboard, per I2's ≤5% target.

These are materially different data-flow and dashboard-design requirements — don't build B9 (or
finish the model input pipeline) until this is settled with the team. Ask directly; don't guess.

---

### 3. Model 1 — N-HiTS (fine-tuned): continuous RUL regression

**Spec**: ICS1 — RUL, MAPE ≤15% *(loosened from an earlier ≤10% target — confirm which is current
before reporting a number)*.

**Mechanism**: multi-rate input sampling lets separate stacks specialize on different frequency
bands of the degradation signal — one stack tracks the slow, multi-day stiffness-decay trend RUL
actually depends on, another absorbs faster load/speed fluctuations. Hierarchical interpolation
reassembles these into one long-horizon forecast.

**Why N-HiTS, not TFT, for this path**: I2/ICS1 must run **continuously** on the always-on server
under the no-cloud constraint (C4), on hardware still being sized between a laptop and a Raspberry
Pi 5 (§6 — unresolved). N-HiTS's own reported benchmark is **~50× lower inference compute than
Transformer baselines at ~20% higher accuracy** — that efficiency, not just algorithm preference,
is why it owns the always-running regression path instead of TFT.

**Evidence**:
- Rank 2 — Challu et al., *AAAI-23* (arXiv:2201.12886): N-HiTS's own reported ~20% average
  accuracy gain over Transformer baselines at ~50× lower compute.
- Rank 2 — team's weighted concept-selection score: Hybrid (N-HiTS+TFT) **4.30** vs. N-HiTS-only
  **4.00** vs. TFT-only **3.40** — hybrid wins on combined RUL+classification coverage, loses only
  on compute efficiency.
- Rank 2 is the honest ceiling for a model-selection decision — no CAD/simulation evidence form
  applies to an algorithm choice.
- Source: `pytorch-forecasting` library (NHiTS confirmed available, rated most compute-efficient
  of the library's 10 models).

**Status**: architecture/library choice justified and evidenced; no fine-tuning on real data has
happened yet (synthetic-data-only stage across the whole project as of FDR).

---

### 4. Model 2 — TFT (fine-tuned): per-window fault classification

**Spec**: ICS2 — fault classification, <200ms/window.

**Mechanism**: recurrent encoder-decoder layers handle local temporal structure; interpretable
multi-head attention captures longer-range dependencies; a variable-selection network separates
static, known-future (e.g. the rig's commanded VFD speed step), and observed-past inputs, so the
classifier needs no hand-engineered feature interactions across the 1000–3600 rpm sweep.

**Why TFT, not N-HiTS, for this path**: ICS2's output must be **trusted by a technician**, not
just accurate — Ch.2.3's ethical analysis flags an over-confident, uninspectable classifier as the
direct false-"healthy"-reading risk (NSPE II.1.a). TFT's attention weights and variable-selection
scores make a fault call **inspectable** rather than a black-box label — that's why TFT, not
N-HiTS, owns per-window classification despite its higher per-call compute cost.

**Evidence**:
- Rank 2 — Lim et al., *International Journal of Forecasting* 37(4), 2021 (arXiv:1912.09363): TFT
  outperforms Amazon's DeepAR by 36–69% across benchmarks.
- Rank 2, analogous published application — Fentaye & Kyprianidis (2024), *Aeronautical Journal*
  128(1325), DOI 10.1017/aer.2024.40 — TFT applied to gas-turbine degradation prognostics on
  rotating machinery, forecasting degradation trend multiple cycles ahead. This is the closest
  real precedent found for TFT on rotating-machinery prognostics specifically.
- Team's weighted score: TFT-only 3.40 — chosen for interpretability, not raw accuracy, per the
  Ch.2.3 ethical requirement.

**Status**: same as N-HiTS — architecture choice justified and evidenced, fine-tuning on real data
still pending.

---

### 5. B9 — .NET Dashboard & Digital Twin Backend

**Spec**: ICS3 — UI refresh ≥10Hz, joint physics+AI+fault update. Shares ownership of I1
(end-to-end latency <500ms) and I2 (see §2's open contradiction) with COE and ME.

**Mechanism**: an ASP.NET Core service ingests the WiFi-streamed feature-window payload from
COE's B6 (the edge node's 168-byte payload — see the COE folder for its exact field layout, and
note the COE working docs currently describe a 152-byte payload, which doesn't match the FDR's
168-byte figure; reconcile with COE before finalizing the parser). Both fine-tuned models run
**in-process** via ONNX Runtime (exported once from the Python training pipeline via
`torch.onnx.export`). Model outputs are combined with the physics-side input (either physics
features or ME's independent physics RUL, per §2) into one payload. A SignalR hub pushes that
payload to every connected browser over WebSockets, refreshing a multi-panel dashboard at ≥10Hz.

**Why this stack**:
- **Kestrel** (ASP.NET Core's built-in server) needs no external service — keeps the
  local-network-only constraint (C5) trivial to enforce.
- **SignalR** is built for exactly this "push the instant it changes" pattern and names dashboards
  as a direct use case.
- Running the ONNX-exported models **in the same process** as the hub removes a network hop before
  the <500ms end-to-end budget (I1).
- **.NET 9** runs natively on the Raspberry Pi 5's ARM64 OS — relevant if the Pi 5 wins the §6
  hardware-sizing decision, though that decision is still open.

**Data flow** (B7–B9): Feature Buffer (32 features + 4 phasors + RPM, one payload per window, from
COE's B6) feeds both the N-HiTS Engine (continuous RUL, §3) and the TFT Engine (per-window
classification, §4) in parallel → the .NET Host Process runs both models in-process via ONNX
Runtime, combines AI RUL + fault class + the physics-side input into one result (exact combination
logic depends on §2's resolution) → SignalR Hub → WebSocket push → Web Dashboard (multi-panel
view, ≥10Hz) + Digital Twin (health index).

**Evidence**:
- Rank 2, vendor docs — SignalR's own use-case list names "company dashboards, instant sales
  updates" directly, WebSockets as its preferred transport (Microsoft Learn, ASP.NET Core SignalR
  docs).
- Rank 2, vendor docs — ONNX Runtime ships an official C#/.NET binding; PyTorch models export to
  it via `torch.onnx.export()` — the documented bridge from the Python-trained models
  (onnxruntime.ai).
- .NET 9 supports Debian 12/13 and Ubuntu 22.04/24.04 on ARM64 (dotnet/core `supported-os.md`) —
  the OS families a Raspberry Pi 5 runs. Vendor-documented compatibility, not yet bench-confirmed
  on the team's own hardware.

**Confirmed resolved (not open)**: the dashboard is a **conventional multi-panel web telemetry
layout**, not a bespoke 2.5D cross-section SVG — the user confirmed "a general web dashboard is
more familiar for users," which is what got designed and built out above.

**Status**: architecture and stack choice justified and evidenced (Rank 2 throughout); no working
implementation yet — dashboard integration is listed as a "Next" item on the FDR's own status
slide, alongside rig assembly, fault-injection data collection, and model fine-tuning.

---

### 6. Hardware this layer depends on — resolved vs. still open

- **Server (this layer's own hardware)**: **still explicitly unresolved.** The FDR itself says the
  always-on server is "hardware still being sized between a laptop and a Raspberry Pi 5." The
  .NET-9-on-Pi5-ARM64 compatibility fact (§5) is a supporting argument if Pi 5 is chosen, not a
  decision that it has been. If a two-path comparison is wanted for the report (laptop vs. Pi 5,
  sizing/cost/power/throughput both calculated), a generic mid-range business laptop was
  previously drafted as the laptop candidate: Intel Core i5 (10-core, "U-series" mobile), 16GB
  RAM, integrated graphics, SSD — deliberately an office/plant-floor spec, not
  gaming/workstation-class, to make the point that no special procurement is needed on that path.
- **Node MCU / edge hardware**: not this layer's responsibility, but this layer's parser depends
  on it — see the COE folder. Resolved to ESP32-C6 (WiFi bridge) + STM32H755 (processing); no
  longer provisional.
- **Network path**: WiFi from the ESP32-C6 bridge to this host, local network only (C5) — no
  cloud exposure of vibration data or dashboard access at any point.

---

### 7. Specifications and constraints this layer owns or shares

| ID | Item | Target | Owner(s) |
|---|---|---|---|
| ICS1 | RUL prediction | MAPE ≤15% — pretrained N-HiTS, fine-tuned | ICS (Jaddoua) |
| ICS2 | Fault classification time | <200 ms/window — pretrained TFT, fine-tuned | ICS (Jaddoua) |
| ICS3 | Dashboard & digital-twin presentation | UI refresh ≥10Hz; joint physics+AI+fault update | ICS (Abdulrazaq) |
| C3 | Market cost ceiling | Node BOM ≤4,000 SAR; sell price ≤8,000 SAR/node | ME + ICS (shared) |
| C5 | Dashboard network access | Local-network-only; no external/cloud exposure | ICS (unique) |
| I1 | End-to-end latency | Sensor → classify+RUL → dashboard <500ms | ME + COE + ICS |
| I2 | **Physics-AI RUL residual ≤5%, stages 1→3** *(or the ablation-study target — see §2)* | | ME + COE + ICS |
| I3 | Classification & stage performance | macro-F1 ≥85%; any fault caught by stage 3 | ME + COE + ICS |

All of ICS's own specs are marked "Met" in the design-stage sense per the FDR's final table — none
are off-the-shelf (this layer is 100% custom software). No prototype data exists yet for any of
them; every "Met" claim is Rank 2 (calculation/design-backed), not measured.

**⚠️ ID scheme note**: these IDs (ICS1–3, C3, C5, I1–3) are the FDR's final numbering. The report
draft still uses an older scheme (I2/I3/I4 for these same three ICS specs, N1–N3 for the
integrated ones) — don't mix the two when writing new material.

---

### 8. Evidence & reporting rules that apply to this layer

- Every "Met (design)" claim above needs the calculation, datasheet, or cited benchmark behind it
  — never a bare assertion. Ranks 2–3 (calculations/predictions, or simulation/CAD) are the honest
  ceiling right now; no prototype exists yet, and that's expected at this stage, not a weakness to
  hide.
- Never claim a test was run, a model was fine-tuned, or a number was measured when it wasn't —
  say plainly what's still planned vs. done.
- State what actually is/was done — never volunteer a limitation nobody asked about (e.g. don't
  write a sentence about *not* having real fine-tuning data yet unless directly relevant to the
  point being made).

---

### 9. Immediate next steps for this layer

1. Get §2's physics-framing contradiction resolved with the team — it decides the actual input
   pipeline and dashboard design for B9.
2. Reconcile the payload size/field mismatch between the COE working docs (152 bytes) and the FDR
   deck (168 bytes) before writing the B9 payload parser.
3. Push for a laptop-vs-Pi5 decision (or build both paths out for the report, per the original
   "show design extent" instruction) — this layer's own deployment target depends on it.
4. Begin real fine-tuning once fault-seeded data exists (currently synthetic-only, ME-owned
   dependency).
5. Second Report Instalment is due 22 Sept 2026 — this layer's Ch.3 model-selection writeup can
   draw directly on §3/§4's evidence, which is richer than what's currently in the report draft.
