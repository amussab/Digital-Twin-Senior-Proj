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

> **Superseded 2026-10-04 by §5.** FEMTO-ST was dropped because its 0.1 s records can't form a
> 20-rev window. CWRU was demoted. MaFaulDa and IMS were added.

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

---

## 5. Dataset selection for pretraining — decided 2026-10-04 (supersedes §3's table)

Re-screened against the rig as it is now actually specified (spec sheet `overview/252-
Specifications-TEAM M001.xlsx`): **2 radial bearings × biaxial (X,Y) IEPE = 4 channels** (Spec 1),
≥25 kS/s (Spec 4, built at 30 kS/s), a **2–8 kHz** bearing-resonance band-pass (COE Component 3),
**20-revolution windows** (Spec 6 / COE C1), steady modes **1750 and 3600 rpm**, 5 classes
{healthy, OD, ID, BD, CF} (Integrated 3), RUL as MAPE (Spec 7). Every dataset has to survive the
same DSP chain (`AI-test/dsp.py`) to land in the 32-feature contract, so these are hard filters,
not preferences.

### 5.1 The key reframing: the common unit is *one biaxial bearing*, not the 4-channel rig

Every strong candidate below has **two orthogonal radial accelerometers per bearing** — exactly the
rig's per-bearing layout. So the "channel-count gap" (CLAUDE.md open item 5) is resolved without
padding or dropping real columns: feature extraction and the models operate **per bearing** on
that bearing's 16 features (X: 8, Y: 8). The rig's 32-feature vector is simply two such bearing
records (indices 0–15 and 16–31) sharing a window index. A public dataset with one test bearing
contributes one series per run; IMS contributes four; the rig contributes two. This also matches
the problem: RUL and fault type are properties of a bearing, and M6 asks to *localize* the fault,
which per-bearing inference gives directly. Cross-bearing context (the current `plane_asymmetry`
input) becomes a rig-only fine-tuning feature, not a pretraining one.

### 5.2 Screening table

| Dataset | Axes per bearing | fs (Nyquist vs 8 kHz) | ≥20 rev per record? | Speed | Labels | Verdict |
|---|---|---|---|---|---|---|
| **XJTU-SY** | 2 (H+V), 1 bearing | 25.6 kHz ✔ | 1.28 s @2100–2400 rpm = 44–51 rev ✔ | 2100/2250/2400 | 15 run-to-failure; failed element per bearing incl. **outer, inner, cage** | **RUL pretraining — primary** |
| **IMS (NASA/Univ. Cincinnati), Test 1** | 2 (orthogonal radial), **4 bearings on one shaft** | 20 kHz ✔ (10 kHz) | 1 s @2000 rpm = 33 rev ✔ | 2000 | Run-to-failure, 34.5 days; B3 inner race, B4 roller failed; snapshots every 10 min (= `SynthConfig` cadence) | **RUL pretraining — secondary**, the only multi-bearing biaxial run-to-failure set |
| **MaFaulDa (UFRJ)** | Underhang: triaxial; overhang: 3 single-axis → keep the 2 radial axes each, drop axial | 50 kHz ✔ | 5 s @737–3686 rpm = 61–307 rev ✔ | **737–3686 rpm sweep, ~60 rpm steps — covers 1750 and 3600** | Seeded bearing faults at both bearings, 3 severities; normal + imbalance + misalignment | **Classification pretraining — primary** |
| KAIST (Jung et al. 2023) | 2 (x,y) on **2 housings** — identical to the rig | 25.6 kHz ✔ | yes | ~3010 rpm; 680–2460 sweep subset | normal, inner, outer (+ misalignment, unbalance); no ball/cage | Classification — supplementary |
| FEMTO-ST / PRONOSTIA | 2 (H+V) | 25.6 kHz ✔ | **0.1 s @1500–1800 rpm = 2.5–3 rev ✘** | 1500/1650/1800 | run-to-failure, no element labels | **Dropped** — a 20-rev window cannot be formed (C4 / COE C1) |
| CWRU | 1 per bearing (DE, FE) | 12 kHz ✘ (6 kHz) for most files; 48 kHz DE only | yes | 1730–1797 | seeded OR/IR/ball, no cage | **Demoted** — single-axis, and the 12 kHz files can't pass a 2–8 kHz band |
| Kumar et al. 2022 pump data (IEEE DataPort 10.21227/8cqr-jr43) | single sensor | n/s | n/s | 2580 rpm (43 Hz) | 4 classes, no cage, no RUL | Methodology precedent only (§1); access-gated |
| Bruinsma et al. 2023 centrifugal-pump data (4TU, CC BY 4.0) | 1 per location (5 single-axis sensors) | 20 kHz ✔ | 60 s ✔ | 735–1470, 2065 rpm | OR/IR/ball ×3 severities + contamination; pump NDE bearing OR only; no cage, no RUL | Not trained on — single-axis. Real **centrifugal pump**, worth citing |

`[SOURCED]` facts above: XJTU-SY — Wang, Lei, Li, Li, *IEEE Trans. Reliability* 69(1), 2020, and
github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets; IMS — Lee, Qiu, Yu, Lin, "Bearing Data Set,"
IMS Univ. of Cincinnati / NASA Prognostics Data Repository, 2007 (Test 1: 2,156 files, Bearing
1 = Ch1&2 … Bearing 4 = Ch7&8, 20,480 pts @20 kHz); MaFaulDa — www02.smt.ufrj.br/~offshore/mfs/
(50 kHz, 5 s, 49 speeds 737–3686 rpm, bearing orders FTF 0.375 / BSF 1.871 / BPFO 2.998 / BPFI
5.002); KAIST — Jung et al., *Data in Brief*, 2023 (PMC10036499); FEMTO — 0.1 s snapshots every
10 s @25.6 kHz (Nectoux et al., PHM 2012); Bruinsma et al., *Data in Brief* 52 (2024) 109987,
DOI 10.1016/j.dib.2023.109987.

### 5.3 Why these two, specifically

- **MaFaulDa's bearing is nearly the rig's bearing.** Its orders (0.375 / 1.871 / 2.998 / 5.002)
  sit within 0.11 order of COE's fixed values (0.38 / 1.98 / 3.05 / 4.95), and both are 8-element
  bearings. The order-family band features (indices 4–7 / 12–15) will therefore look at almost
  the same order bins in pretraining and on the rig — the closest feature-space match of any public
  dataset found. Its speed sweep spans both of the rig's steady modes, which is what C4's
  speed-independent representation has to hold up against.
- **XJTU-SY is the only sizeable biaxial run-to-failure set whose records are long enough** for a
  20-rev window, and it labels the failed element, including cage — the class CWRU lacks entirely.
  IMS Test 1 adds a multi-bearing shaft and the same 10-min cadence the synthetic generator
  assumes, but it is a single experiment, so it cross-checks rather than carries the RUL model.
- **⚠️ Verify on download, not yet confirmed**: MaFaulDa's own page lists the bearing defect
  types inconsistently (its tables say outer track / rolling elements / inner track; section
  headings and most papers say outer race / ball / cage). The actual directory names decide which
  classes it supplies. If it has no inner-race class, inner race comes from XJTU-SY, IMS B3, and
  KAIST.

### 5.4 Consequences for the pipeline (not yet implemented)

1. **RUL target must be scale-invariant** (percent of life remaining). XJTU-SY lives span roughly
   0.7 h to 42 h at 1-min snapshots, IMS is 34 days at 10-min snapshots. Pretraining across those
   in absolute hours is incoherent. This is the same choice as ICS1 option 1 in
   `AI-test/README.md`, now forced by the data. **It still needs the team's sign-off on Spec 7's
   wording.**
2. **Time axis differs per source** (1 min vs 10 min vs rig cadence). The encoder sees a
   window index, so pretraining should resample each run to a common number of windows per unit of
   life fraction, or explicitly treat cadence as a known input. Decide this before training.
3. **Dimension reduction**: the per-bearing unit removes the need to *drop* real columns. PCA
   is possible as an added step, but it has two costs measured against the specs. First, a PCA
   basis is fitted on source-domain variance, so it moves under domain shift — the exact risk in
   §4. Second, it replaces named inputs with components, which defeats TFT's variable-selection
   interpretability, the stated reason TFT owns ICS2 (Ch.2.3 ethics). Recommendation: keep the
   named-feature reduction already in `features.py`, and run PCA only as a reported ablation if
   the instructors want it shown. If it is used, its matrix must go into `model_contract.json`
   (fitted state, not a code constant).
4. **Kumar et al. §1 numbers — correction**: the paper's own text gives SEDA as 99.55 % source /
   24.77 % target, while its Table 3 total column reads 99.23. The paper is internally
   inconsistent; quote the text figure and note the discrepancy if it is cited.
