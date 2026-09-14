# Real-Time Edge AI Bearing Monitoring System
## ICS — External Datasets & Literature Precedent

Extracted 2026-09-14, supporting `ICS AI, Dashboard & Digital Twin Layer.md` in this same folder.
Covers one paper found in the project's `resources/` folder and three externally-sourced
benchmark datasets, checked for relevance to ICS1 (N-HiTS RUL regression) and ICS2 (TFT fault
classification), and independently verified where a URL or fact could be checked. Corrections to
what was originally supplied are called out explicitly, not silently fixed.

---

## 1. The paper in `resources/` — directly relevant, worth citing in Ch.3

**Kumar, A., Kumar, R., Xiang, J., Qiao, Z., Zhou, Y., Shao, H.** "Digital twin-assisted AI
framework based on domain adaptation for bearing defect diagnosis in the centrifugal pump."
*Measurement*, 235 (2024) 115013. DOI: [10.1016/j.measurement.2024.115013](https://doi.org/10.1016/j.measurement.2024.115013).

**Why this is a strong match, not just a loosely-related citation**: this is one of the only
published papers that combines *all three* of the project's defining elements at once — a digital
twin, AI-based diagnosis, and a **centrifugal pump** specifically (not a generic bearing test rig).
The team's own architecture (physics model generating synthetic training data + AI models handling
the real diagnosis/prediction task) is structurally the same idea this paper implements and
validates.

**What it does**:
- Builds a lumped-mass dynamic simulation of the pump's rotor-bearing-impeller system (bearings
  as spring-damper elements) — mechanically the same category of physics model as this project's
  FE beam model in `ME/`, though theirs uses a lumped-mass formulation instead of an FE beam.
- Uses that simulation to generate **synthetic vibration data** for 4 conditions (defect-free,
  inner race, outer race, roller/ball defect) — directly addresses the same labeled-data-scarcity
  problem this project has (synthetic-only data as of FDR, real fault-seeded data not yet
  collected).
- Runs a **1D CNN classifier with a domain-adversarial branch** (Gradient Reversal Layer) to
  transfer what it learns from the synthetic ("source domain") data to a small set of **real**
  experimental vibration data ("target domain") from an actual test rig (test bearing: 6203-ZZ).
- Compares three domain-adaptation methods — DANN, Maximum Classifier Discrepancy (MCD), and
  Self-Ensembling Domain Adaptation (SEDA) — and reports real numbers: **DAN reached 89.27% source
  / 99.55% target accuracy; MCD 83.26% / 100%; SEDA 99.23% / 24.77%** (SEDA overfit badly to the
  source domain and failed to generalize — a genuinely useful cautionary result, not just a
  positive one).

**Where this is directly actionable for ICS1/ICS2, not just background reading**: this project
currently has no explicit plan for the gap between "trained on synthetic data" and "deployed on
real pump vibration" — the FDR's own evidence-status lines call out real fault-seeded data as
"not yet collected" with no stated bridging strategy. This paper *is* a bridging strategy: instead
of hoping synthetic-trained N-HiTS/TFT models generalize to the real rig by default, a
domain-adversarial fine-tuning step (even a lightweight one — the model in this paper is a single
Conv1D layer + pooling + dense layers, not large) could be proposed as a concrete answer to that
gap in Ch.3's design discussion. This would also give TFT's fault-classification path (ICS2) a
directly analogous, pump-specific published precedent to cite — stronger than the currently-cited
Fentaye & Kyprianidis (2024) TFT paper, which is about gas-turbine prognostics, not pump bearings.

**The paper's own dataset — checked, real, but access-gated**: the paper's Data Availability
statement points to an IEEE DataPort entry (DOI 10.21227/8cqr-jr43). Verified: this resolves to
**"Acoustic and vibration data for defect cases of the centrifugal pump"** (Anil Kumar & Rajesh
Kumar, 2022, Sant Longowal Institute of Engineering and Technology). It is real pump-bearing
acoustic + vibration data for the same 4 defect classes — the closest real-world match to this
project's own target machine found anywhere in this search. **Caveat**: the IEEE DataPort listing
doesn't show a plain download button — it directs researchers to email the author
(anil_taneja86@yahoo.com) for access, and also requires citing four of the author's associated
papers. Worth requesting given the fit, but budget for response-time uncertainty — don't plan
around having it in hand by a specific deadline.

**Honesty note for whichever section cites this**: the paper's own experimental "target domain" is
still a lab bench-rig bearing (6203-ZZ), not literally the team's own RK4 kit or its bearing model
— cite it as design/methodology precedent (Rank 2, analogous published application — same evidence
tier as the Fentaye & Kyprianidis TFT citation already in `ICS AI, Dashboard & Digital Twin
Layer.md`), not as validation of this project's own numbers.

---

## 2. External benchmark datasets

The three datasets below were proposed for pretraining/validating the model pipeline before real
rig data exists. All three were checked: dataset existence, official source legitimacy, and the
core facts (bearing count, conditions, seeded-fault vs. run-to-failure). **Two of the three
secondary/mirror URLs originally supplied were wrong and are corrected below** — the primary/
official sources were correct as given.

### 2.1 CWRU (Case Western Reserve University) Bearing Data Center — for fault classification (ICS2/TFT)

**Confirmed accurate as supplied.** Seeded faults (outer race, inner race, ball) at multiple fault
diameters — **not** run-to-failure, so this trains/validates the classification head (ICS2), not
RUL regression (ICS1).

- Official source: <https://engineering.case.edu/bearingdatacenter/download-data-file> — verified
  live; hosts Normal Baseline, 12k/48k Drive-End, and Fan-End fault data as MATLAB files.
- Easier access: `github.com/Litchiware/cwru` / `pip install cwru` — verified to exist. One caveat
  found: the repo has open issues and at least one merged fix for download-URL breakage and
  Python-3 compatibility, meaning it's needed patching before as CWRU's own site changed — smoke-
  test it before relying on it as a turnkey install, don't assume `pip install cwru` will just
  work unmodified.

**Bearing size note**: CWRU's bearings are SKF/other standard deep-groove ball bearings in the
62xx/63xx families at specific fault-diameter increments — not literally the RK4 rig's bearing
model. Same caveat as everywhere else this shows up: legitimate for pretraining pipeline mechanics,
not a stand-in for the project's own hardware.

### 2.2 FEMTO-ST / PRONOSTIA (IEEE PHM 2012 Prognostic Challenge) — for RUL regression (ICS1/N-HiTS)

**Core facts confirmed, one URL corrected.** Real run-to-failure accelerated-degradation data: 6
training + 11 test bearings = **17 bearings total**, **3 operating conditions**, lifespans roughly
1–7 hours — this is the dataset in this list that actually has genuine time-to-failure labels,
which is exactly what ICS1's RUL regression needs and currently lacks (the project's own RUL
labels are synthetic, from an ISO-281-style life law, not measured degradation).

- Origin/description: <https://publiweb.femto-st.fr/tntnet/entries/1528/documents/author/data> —
  verified reachable; this is the Nectoux et al. PRONOSTIA platform description page.
- **⚠️ Correction**: the originally-supplied IEEE DataPort link (`ieee-dataport.org/node/1849`) does
  **not** point to this dataset — it actually resolves to an unrelated "CWRU bearing dataset and
  Gearbox dataset of IEEE PHM Challenge Competition 2009" listing. Drop that link.
- Confirmed true: FEMTO-ST's own direct hosting is inconsistently available — a mirror repo's
  README explicitly states the original femto-st.fr data URL "used to be online... but isn't
  anymore." **Working alternative**: the GitHub mirror `github.com/wkzs111/phm-ieee-2012-data-
  challenge-dataset` (forks also exist under `AaronCosmos` and `Lucky-Loek`) — use this instead of
  the IEEE DataPort node.

### 2.3 XJTU-SY — backup/cross-validation dataset for RUL (ICS1/N-HiTS)

**Core facts confirmed, one URL corrected.** Real run-to-failure data, **15 bearings**, **3
conditions** (2100rpm/12kN, 2250rpm/11kN, 2400rpm/10kN), from Xi'an Jiaotong University +
Changxing Sumyoung Technology. Legitimate as a second RUL dataset to check a model isn't overfit
to FEMTO-ST's specific rig/conditions.

- Description: the `scholar.xjtu.edu.cn` tutorial page exists as supplied.
- **⚠️ Correction**: the originally-supplied GitHub mirror (`github.com/cathysiyu/Mechanical-
  datasets`) does **not** contain XJTU-SY data — its actual contents are CWRU bearing data plus a
  Southeast University gearbox dataset (i.e. it's a real repo, just the wrong one for this
  purpose). **Correct repo**: `github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets` — the
  author-maintained source, with official mirrors (Google Drive/Dropbox/MediaFire/MEGA/Baidu
  links).
- Processed/ready-to-use version on Mendeley Data — confirmed to exist and match ("Bearing_Dataset_
  XJTU_Processed", ID `mpn45f4gxc`) at <https://data.mendeley.com/datasets/mpn45f4gxc>; direct
  fetch hit an anti-bot block during this check, but title/ID were independently confirmed, so
  treat it as legitimate rather than dead.

---

## 3. How these fit the project, concretely

| Purpose | Best fit | Why |
|---|---|---|
| ICS1 (N-HiTS RUL) pretraining | FEMTO-ST/PRONOSTIA, primary | Only real dataset here with actual time-to-failure labels |
| ICS1 cross-validation | XJTU-SY | Same category, different rig — checks overfitting to one dataset's quirks |
| ICS2 (TFT classification) pretraining | CWRU | Standard, well-documented, labeled by fault type/location |
| ICS2 methodology precedent, and a possible answer to the synthetic→real gap | Kumar et al. 2024 (§1) | Only source here that's actually about a centrifugal pump, and the only one with a concrete domain-adaptation strategy for bridging simulated and real data |
| Closest real-world data to this project's own target machine | Kumar et al.'s own dataset (DOI 10.21227/8cqr-jr43) | Real pump acoustic+vibration data, same 4 defect classes — but access-gated, email the author |

**Standing caveat, same as everywhere else in this project's evidence**: none of the four datasets
above are this project's own rig or its own bearing. They're legitimate for pretraining/validating
pipeline mechanics and for citing published precedent — not for reporting an RUL or accuracy
number as if it describes this project's own hardware. Keep that distinction explicit wherever any
of this shows up in the report or slides, exactly like the "synthetic data" labeling used
elsewhere in this project's evidence.

---

## 4. Risk check — "pretrain on public data, then fine-tune on our own rig data"

This is the team's actual stated plan (as of 2026-09-14) and it's the right overall framework — it's
literally why N-HiTS/TFT were chosen as *pretrained, fine-tuned* models rather than trained from
scratch. But "fine-tune" is doing a lot of unstated work in that plan, and the Kumar et al. 2024
paper (§1) has direct evidence that the naive version of this — just continuing training on the
new data without addressing the domain gap — can fail outright rather than just underperform. Its
own numbers, same task, two methods: DAN went from 89.27% (synthetic/source) to 99.55%
(real/target) accuracy; SEDA went from 99.23% down to 24.77% doing the same synthetic→real switch.
Same problem, opposite outcomes, purely on how the transfer was handled. Three concrete risks to
close before this is a real plan rather than a one-line intention:

1. **Feature-space mismatch.** N-HiTS/TFT are fed this project's specific 32-value envelope +
   order-FFT feature set, computed for this project's own bearing geometry and window scheme. CWRU/
   FEMTO-ST/XJTU-SY are raw vibration signals from different bearings at different sample rates —
   they are not usable as pretraining input until they're run through the *same* feature-extraction
   pipeline the rig's own data will use. That reprocessing step is real engineering work and isn't
   currently scoped anywhere in the project plan.
2. **RUL label scale mismatch.** FEMTO-ST/XJTU-SY's RUL labels are absolute hours-to-failure for
   their own accelerated-life tests; this project's own labels come from an ISO-281-style life law
   on a different physics model. Pretraining on one scale and fine-tuning on another, without
   normalizing to something scale-invariant (e.g. percent-of-life-remaining instead of absolute
   time), risks a model that's technically working but consistently miscalibrated in a way that's
   easy to miss during development.
3. **Small fine-tuning set makes this worse, not less important.** The rig-assembly and
   fault-injection data collection (FDR's own "Next" list) will happen in the same short window as
   everything else — the real fine-tuning dataset is likely to be small. Small target-domain data is
   exactly the regime where naive continued training is riskiest and where a domain-adaptation-aware
   approach (freeze/adapt the feature extractor, add a domain-adversarial branch — the DANN approach
   Kumar et al. use, not full retraining) earns its keep.

**Recommendation for the report/Ch.3**: don't leave "we fine-tune on our own data" as the full
explanation. State the actual mechanism — feature-space alignment before pretraining, label
normalization, and (if the domain gap proves large in practice) a domain-adaptation step — and cite
Kumar et al. 2024's DAN-vs-SEDA result as the evidence for why the mechanism matters, not just the
end goal. This is also a stronger answer to the kind of scrutiny Al-Badour has already shown he
applies to under-specified claims.
