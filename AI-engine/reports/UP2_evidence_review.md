# UP2 evidence review (evidence-reviewer, 2026-10-24)

Scope: `evidence/UP2_Evidence.md`, `notebooks/UP2_Evidence.ipynb` (+ `up2_helpers.py`), `evidence/up2_figures/*`, spec-check, residual, COE and integration reports. Ladder: `.claude/rules/evidence-strength.md`. Read-only review; nothing was fixed.

## Verdict
The honesty of the package is good: synthetic numbers are tagged, the public-data FAILs (S7 281.8 %, IS3 F1 0.581) sit beside the new numbers, the 2nd-look and 3rd-look test disclosures are present, IS2 is reported as FAIL. The **counts and ranks are generous**. The claim "10/18 MET, every row Rank 9" does not survive the ladder. Defensible: **7 met with firm evidence, 10 only if simulated/calculated hardware legs are accepted as "met (design)". 14/18 is not reachable** from ICS+COE alone (7 ME/COE items have no artifact, IS2 fails). Dept block (C5 + S7 + S8): C5 and S8 firm; S7 is met only if the model-name wording is waived.

## 1. Independent recomputation (fresh `synth_v3_test2`, saved v3b predictions, own pandas code, truth regenerated from the generator)
| Metric | Claimed | Recomputed | Note |
|---|---|---|---|
| S7 MAPE (pre-registered window) | 12.64 % | **12.64 %** (own window re-implementation 12.40 % bearing-avg, 12.68 % pooled) | 80 bearings; 75 % of bearings <=15 %, worst 26.4 % |
| S7 true-onset variant | 14.72 % | 14.72 % | margin 0.28; same variant on the old test = 17.44 % (FAIL) |
| IS3a macro-F1 | 0.9587 | 0.9587 (acc 0.976) | truth = onset-based labels, not detected |
| IS3b stage error (true_stage, 80 run-to-failure bearings) | max 1 | max 1, 100 % within 1 | |
| IS3c caught by stage 3 | 80/80 | 80/80 (79 at stage 2, 1 at stage 1) | trivially early: stage 2 starts at onset |
| S8 classifier p95 | 0.04 ms (log) / 0.315 ms (master table) | 0.054 ms (ONNX, 1 thread) | the two documents disagree; both pass by >600x |
| Tests | 36 pass (ledger) | `pytest tests/coe tests/up2`: **45 passed** | dotnet suite not re-run |
| Splits | disjoint, registered first | test2 (200 units) and val2 (240) share no unit with synth_v3 train/val/test; `splits.json` registered 06:13Z (09:13+03, commit 0fdb880); first v3b fit commit 400c8ed 14:40+03; scoring c60c878 14:47+03; seeds 505/606/707 and run offsets distinct | leak-free as far as ID and seed go |

## 2. Per-item grading (claimed rank 9 for every MET row; justified rank per ladder)
| Item | Claimed | Justified | Tag ok? | Issue |
|---|---|---|---|---|
| C1,C2,C3,C6,S1,S2,S3 | not evidenced | 0 | yes | honest; owners must supply. S2 precursor correctly not counted |
| C4 no cloud | MET 9 | 7-9 | yes | network-denied run covers Python inference, not the C# backend path; acceptable |
| C5 LAN only | MET 9 | 9 | yes | 19 live tests; strongest ICS item |
| S4 >=25 kS/s | MET 9 | **3** | partly | "gap-free" simulation cannot fail: the source emits whatever rate it is told; 16.3 us slack is arithmetic on assumed ADC timing |
| S5 <=333 ms | MET 9 | **3-6** | yes | host 7.5 ms measured is Python on a PC; STM32 28 ms is a calculation with an assumed core/clock; margin 11.7x makes "met (design)" fair |
| S6 fixed input | MET 9 | 6 | yes | sim over 28 speeds is fine, but the COE report notes envelope truncation below ~1465 rpm; this caveat is absent from UP2_Evidence.md |
| S7 RUL | MET 9 | **6** | yes | see 3 |
| S8 latency | MET 9 | 6-9 | yes | holds for the spec-named TFT too (1.54 ms), so this one is safe |
| S9 >=10 Hz | MET 9 | **6-7** | partly | 30 Hz is a client-side count of a 50 ms **re-broadcast timer**; new data arrives at the 10-20 Hz replay rate; SignalR client, not browser render; spec also says physics RUL updated together; approval column is still "No" |
| IS1 <500 ms | MET 9 | **6** | yes | Wi-Fi hop is injected delay, STM32 leg calculated; margin is 4.6x so a sensitivity line (Wi-Fi up to 300 ms still passes) would harden it |
| IS2 | NOT MET 6 | 6 | yes | honest; circular stiffness trajectory disclosed |
| IS3 | MET 9 | **6** | yes | see 3 |

## 3. Overclaims and risks
1. **Spec names the models.** S7 reads "pretrained NHiTS, fine-tuned", S8/IS3 "pretrained TFT". The spec-named models **fail** (N-HiTS S7 46.9 %, TFT stage error 3). The MET rows rest on gradient-boosted trees in the same ONNX slots. It is disclosed in section 5 of the summary but not in the status table or the master-table "Measured" column. A grader reading the spec literally gets S7 = FAIL.
2. **Rank 9 on S7/IS3 is wrong**: the notebook re-scores stored predictions (rule 2, rank 6-8); no model is run live on synthetic input in front of the reviewer. Data are in-distribution synthetic (magnitudes calibrated to XJTU-SY, knob ranges narrowed on validation), so a pass shows the pipeline works, nothing more.
3. **S7 definition flatters.** Window starts at *detected* onset and drops the last 10 % of the span; the true-onset variant passes by 0.28 points on test2 and fails (17.44 %) on the old test; the percent-of-life variant (5.8 pts) is a different quantity. Reported alongside, not substituted, which is acceptable, but "12.64 %" should always be quoted with "pre-registered window, synthetic".
4. **Design iteration saw the old test.** v3 failed S7/IS3b on the first test set; v3b's post-process and stage head were then redesigned. Parameters were chosen on validation only and the verdict is on the fresh test2 (good), but the old-test "2nd look" and the later look at test2 by `up2_eval` re-scoring should be counted honestly: test2 was predicted once, re-scored read-only.
5. **Domain gap breaks the end-to-end story.** Models are trained on payload-level synthetic features, not COE-pipeline output; the COE simulator demo misclassifies bearing 2. So IS1/IS3 are not shown on one connected chain with the same data. Disclosed in one bullet; the demo itself should show or state it.
6. **Healthy bearings sit at stage 2** (99.8 % of 45k healthy-companion windows predicted stage 2 against true stage 1). IS3b scores run-to-failure bearings only, so this is invisible in the metric; a grader could ask.
7. Inconsistent S8 numbers across documents (0.04 / 0.054 / 0.3 ms). Cosmetic, fix before the demo.

## 4. Fixes, by value
1. Add a status-table column "model actually deployed" and show the TFT/N-HiTS FAILs on the same row as the GBM PASS; decide with the coach whether the spec wording changes to "gradient-boosted/learned model" (a spec change, not a quiet redefinition). Highest risk for the 50-point department block.
2. Re-rank: S4 3, S5 3-6, IS1 6, S7/IS3 6, S9 6-7; report 7 firm + 3 design-level, not "10 at Rank 9".
3. Make one cell run live: load the ONNX heads, push 120 synthetic windows, print MAPE/F1 (turns 6 into 9 for S7/S8/IS3).
4. S9: measure distinct-data refresh (count of unique sequence numbers per second) and render in the browser; show physics RUL on the dashboard or drop it from the claim.
5. IS1: add a sensitivity row (Wi-Fi hop 100/300 ms) so the margin is explicit.
6. State the S6 truncation below ~1465 rpm and the healthy-at-stage-2 behaviour in the summary.
7. Chase the 7 unevidenced items; each is worth the same as an ICS item, and 4 more are needed for the 75 % bar.
