# ICS AI testbench -- specification check

Generated 2026-09-14T180439Z (UTC)

> **These results are measured on SYNTHETIC data.** No fault-seeded or
> run-to-failure data from the team's own rig exists yet. The numbers
> below verify that the pipeline, the models and the latency path
> behave as designed; they are NOT measurements of the system's
> accuracy on a real bearing and must not be quoted as such. Models
> are scored on held-out runs they never saw in training.


## Specification verdicts

| Spec | Requirement | Target | Measured | Verdict |
|---|---|---|---|---|
| ICS1 | RUL prediction accuracy (N-HiTS) | <= 15.0 % | 38.4 % | **FAIL** |
| ICS2 | Fault classification latency (TFT) | <= 200.0 ms | 11.5 ms | **PASS** |
| I3a | Classification performance | >= 0.850 | 0.980 | **PASS** |
| I3b | Health-stage estimation error | >= 0.950 | 1.000 | **PASS** |
| I3c | Early detection | >= 1.000 | 1.000 | **PASS** |
| I1 | ICS share of end-to-end latency | <= 500.0 ms | 16.4 ms | **PASS** |

### Notes on the verdicts above

- **ICS1** -- Loosened from an earlier <=10% target; confirm which is current before quoting a number in the report.
- **ICS2** -- Measured per 20-revolution window, single-window batch.
- **I3a** -- macro-F1 across the five fault classes.
- **I3b** -- Fraction of windows whose predicted stage is within +/-1 stage. I3 states 'stage error <=1'; 0.95 is this testbench's chosen pass threshold for that, not a value from the spec table.
- **I3c** -- Fraction of faulted runs whose fault is correctly and stably called no later than health stage 3.
- **I1** -- I1's full <500ms budget spans sensor->dashboard. This measures only the ICS inference leg (both models on one window); the acquisition, DSP and network legs are COE-owned and not included here.

## ICS1 -- RUL detail

- Blended estimator MAPE: **38.4 %**
- Population-prior-only baseline MAPE: 44.1 % (what the system would score with no N-HiTS forecast at all)
- Mean absolute error: 10.5 h
- Evaluated windows: 2,868

**Information floor: 31.8% MAPE.** Across the faulted test runs, total life varies by 2.5x. The health index reveals how far through its life a bearing is, not how long that life is; since RUL = total_life x (1 - fraction_consumed), the relative error in RUL can be no better than the relative error in total life. No forecaster of any quality beats this bound while RUL is expressed in absolute hours. Read the measured MAPE against this floor, not against zero.

- Scale-invariant alternative -- percent-of-life-remaining error: 15.9 percentage points. This is the metric the external-datasets doc recommends for cross-dataset work (risk 2), and it is not subject to the floor above.

MAPE by true health stage:

| Stage | MAPE |
|---|---|
| 1 | 32.6 % |
| 2 | 38.6 % |
| 3 | 33.8 % |
| 4 | 40.8 % |
| 5 | 60.6 % |
| 6 | 56.4 % |

## ICS2 / I3 -- classification detail

- macro-F1: **0.980**
- Accuracy: 0.978
- Missed-fault rate (faulted window called healthy): 0.012
- False-alarm rate (healthy window called faulted): 0.041
- Faults caught by stage 3: 1.000 across 8 faulted test runs (median catch at stage 2)

```
              precision    recall  f1-score   support

     healthy      0.978     0.959     0.968      1434
  outer_race      0.951     0.999     0.974       754
  inner_race      0.983     0.994     0.988       793
        ball      1.000     0.983     0.991       528
        cage      0.988     0.969     0.979       524

    accuracy                          0.978      4033
   macro avg      0.980     0.981     0.980      4033
weighted avg      0.978     0.978     0.978      4033
```

Confusion matrix (rows = true, columns = predicted):

| | healthy | outer_race | inner_race | ball | cage |
|---|---|---|---|---|---|
| **healthy** | 1375 | 39 | 14 | 0 | 6 |
| **outer_race** | 1 | 753 | 0 | 0 | 0 |
| **inner_race** | 5 | 0 | 788 | 0 | 0 |
| **ball** | 9 | 0 | 0 | 519 | 0 |
| **cage** | 16 | 0 | 0 | 0 | 508 |

## Latency detail

Single-window inference, batch size 1, the path the deployed host runs.

| Path | mean | p50 | p95 | max |
|---|---|---|---|---|
| TFT classification | 7.0 ms | 5.1 ms | 11.5 ms | 13.2 ms |
| N-HiTS forecast | 2.8 ms | 2.2 ms | 5.0 ms | 6.5 ms |

Measured on: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel / Windows-10-10.0.26200-SP0, PyTorch 2.14.0+cpu, CPU.

> The deployment target is still open between a laptop and a Raspberry Pi 5 (ICS layer doc, section 6). These timings are from the development machine and do not stand in for either candidate. Re-run this on the chosen hardware before quoting a latency figure.
