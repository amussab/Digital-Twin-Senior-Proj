# AI-engine spec check (20261004T192939Z)

> **SYNTHETIC DATA ONLY.** Every number below comes from `synthetic_rig`, generated data. It proves the pipeline runs end to end; it is NOT a measurement of any real bearing and must not be quoted as one.

- Seed `20261004`, git `2b7a1cd` (+uncommitted changes)
- Command: `app.py evaluate --datasets synthetic_rig --run synthetic --tag synthetic`
- Run dir: `synthetic`
- Host: Windows 10, Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, torch 2.14.0+cpu

## Verdicts

| Spec | Requirement | Target | Measured | Verdict |
|---|---|---|---|---|
| S7 | RUL MAPE on held-out degradation window (N-HiTS) | <= 15.0 % | 18.9 % | **FAIL** |
| S8 | TFT classification latency per window, p95 single-window | < 200.0 ms | 1.7 ms | **PASS** |
| IS3a | macro-F1 over 5 classes (test windows) | >= 0.85  | 0.860 | **PASS** |
| IS3b | stage error <= 1 (fraction of windows, target 0.95) | >= 0.95  | 1.000 | **PASS** |
| IS3c | run-to-failure faults caught (3 consecutive correct) by stage 3 | >= 1.0  | 1.000 | **PASS** |
| IS1 (ICS leg) | features + TFT + N-HiTS + RUL per window, p95 | < 500.0 ms | 2.3 ms | **PASS** |

## S7 -- RUL detail (pre-registered metric, DESIGN.md section 7)

- S7 MAPE (bearing-averaged, degradation window minus last 10%): **18.9 %**
- MAPE over the full degradation window: 24.5 %
- RMSE: 10.0 h; PHM-2012 score: 0.502; percent-of-life error: 8.9 pts
- Estimator components: trend-only 31.5 %, onset-only 145.3 %
- Stage: within +/-1 1.000, mean abs error 0.122, max 1

| Test bearing | MAPE (S7 window) | MAPE (full window) |
|---|---|---|
| synthetic_rig:run000:B1 | 21.4 % | 23.5 % |
| synthetic_rig:run003:B2 | 22.0 % | 21.5 % |
| synthetic_rig:run009:B1 | 11.6 % | 16.8 % |
| synthetic_rig:run032:B1 | 18.7 % | 22.5 % |
| synthetic_rig:run033:B2 | 19.7 % | 23.6 % |
| synthetic_rig:run035:B1 | 17.9 % | 22.2 % |
| synthetic_rig:run042:B1 | 25.2 % | 41.9 % |
| synthetic_rig:run047:B1 | 14.4 % | 24.0 % |

## IS3 -- classification

- Test windows: 8888; macro-F1 over classes present ['healthy', 'outer_race', 'inner_race', 'ball', 'cage']: **0.860** (over all 5 incl. absent: 0.860)
- Missed-fault rate 0.033, false-alarm rate 0.139

| Class | F1 | Support |
|---|---|---|
| healthy | 0.918 | 6209 |
| outer_race | 0.850 | 740 |
| inner_race | 0.799 | 517 |
| ball | 0.837 | 795 |
| cage | 0.896 | 627 |

Confusion (rows = true, cols = predicted, order healthy, outer_race, inner_race, ball, cage):

```
    healthy    5348     243     255     271      92
 outer_race      14     726       0       0       0
 inner_race       3       0     514       0       0
       ball      28       0       0     767       0
       cage      44       0       0       0     583
```

Per dataset (source-confounding check):

| Dataset | n | classes | macro-F1 | recall per class |
|---|---|---|---|---|
| synthetic_rig | 8888 | 5 | 0.860 | healthy 0.86, outer_race 0.98, inner_race 0.99, ball 0.96, cage 0.93 |

Training-class sources (windows per class per dataset, train split):

```
{
 "ball": {
  "synthetic_rig": 1692
 },
 "cage": {
  "synthetic_rig": 1280
 },
 "healthy": {
  "synthetic_rig": 16094
 },
 "inner_race": {
  "synthetic_rig": 1971
 },
 "outer_race": {
  "synthetic_rig": 2531
 }
}
```

## Detection

- Run-to-failure test faults caught by stage 3: 1.000 over 8 unit(s)
- Seeded-fault test records called correctly (3 consecutive windows): n/a over 0 record(s)

## Latency (single window, batch 1, 1 thread)

| Path | p50 | p95 | max |
|---|---|---|---|
| tft_torch | 6.13 ms | 7.26 ms | 10.31 ms |
| tft_onnx | 1.41 ms | 1.66 ms | 2.31 ms |
| nhits_torch | 1.48 ms | 1.87 ms | 2.70 ms |
| nhits_onnx | 0.06 ms | 0.07 ms | 0.15 ms |
| engine_step | 1.88 ms | 2.27 ms | 2.94 ms |

## Track: pretrain_test_only_rul

```
{
 "S7_mape_pct": 18.853776123113505,
 "per_bearing_mape_pct": {
  "synthetic_rig:run000:B1": 21.373928871769987,
  "synthetic_rig:run003:B2": 22.040834132586074,
  "synthetic_rig:run009:B1": 11.559226527197334,
  "synthetic_rig:run032:B1": 18.708820787757237,
  "synthetic_rig:run033:B2": 19.739984788002577,
  "synthetic_rig:run035:B1": 17.87943518312357,
  "synthetic_rig:run042:B1": 25.17243704084801,
  "synthetic_rig:run047:B1": 14.355541653623243
 }
}
```

## Counts

```
{
 "synthetic_rig": {
  "test": {
   "units": 20,
   "windows": 8928
  },
  "train": {
   "units": 60,
   "windows": 23568
  },
  "val": {
   "units": 16,
   "windows": 6970
  }
 }
}
```

Full JSON: `reports/spec_check_20261004T192939Z_synthetic.json`
