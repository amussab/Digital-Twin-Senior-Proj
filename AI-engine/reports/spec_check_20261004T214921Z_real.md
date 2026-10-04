# AI-engine spec check (20261004T214921Z)

> **MEASURED on XJTU-SY / IMS public bearing datasets, not the team rig.** [MEASURED on xjtu_sy, ims] -- scored on held-out test bearings never used for training, early stopping, model selection or calibration. These are measurements of THOSE datasets' bearings; no rig data exists yet.

- Seed `20261004`, git `170bd1c` (+uncommitted changes)
- Command: `app.py evaluate --datasets xjtu_sy ims --run real --tag real --robustness --threads 8`
- Run dir: `real`
- Host: Windows 10, Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, torch 2.14.0+cpu

## Verdicts

| Spec | Requirement | Target | Measured | Verdict |
|---|---|---|---|---|
| S7 | RUL MAPE on held-out degradation window (N-HiTS) | <= 15.0 % | 259.5 % | **FAIL** |
| S8 | TFT classification latency per window, p95 single-window | < 200.0 ms | 11.4 ms | **PASS** |
| IS3a | macro-F1 over the testable classes (train AND test support; primary) | >= 0.85  | 0.566 | **FAIL** |
| IS3a-5 | macro-F1 over all 5 classes (ball has no training bearing -> F1 0) | >= 0.85  | 0.288 | **FAIL** |
| IS3b | stage error <= 1, literal reading: MAX abs stage error over test windows | <= 1  | 4.000 | **FAIL** |
| IS3c | run-to-failure faults caught (3 consecutive correct) by stage 3 | >= 1.0  | 0.500 | **FAIL** |
| IS1 (ICS leg) | features + TFT + N-HiTS + RUL per window, p95 | < 500.0 ms | 14.8 ms | **PASS** |

## S7 -- RUL detail (pre-registered metric, DESIGN.md section 7)

- S7 MAPE (bearing-averaged, degradation window minus last 10%): **259.5 %**
- MAPE over the full degradation window: 604.8 %
- RMSE: 1010.3 h; PHM-2012 score: 0.048; percent-of-life error: 93.9 pts
- Estimator components: trend-only 588.5 %, onset-only 941.2 %
- Stage (IS3b): max abs error 4, mean abs error 0.899, fraction within +/-1 0.721 over 4132 windows

| Test bearing | MAPE (S7 window) | MAPE (full window) |
|---|---|---|
| ims:test1:B4 | 468.6 % | 1093.7 % |
| xjtu_sy:Bearing1_4 | 464.2 % | 1014.5 % |
| xjtu_sy:Bearing2_5 | 50.9 % | 177.0 % |
| xjtu_sy:Bearing3_4 | 54.5 % | 133.8 % |

## IS3 -- classification

- Test windows: 4124. **Primary IS3a macro-F1 over the testable classes ['healthy', 'outer_race', 'inner_race', 'cage'] (2137 windows): 0.566**
- macro-F1 over all 5 classes: 0.288. Classes the model was trained on: ['healthy', 'outer_race', 'inner_race', 'cage']. Not testable: {'ball': 'test support 1987 windows, 0 training support (no output channel)'}
- Missed-fault rate 0.678, false-alarm rate 0.001

| Class | F1 | Support |
|---|---|---|
| healthy | 0.690 | 1777 |
| outer_race | 0.523 | 219 |
| inner_race | 0.213 | 98 |
| ball | 0.000 | 1987 |
| cage | 0.014 | 43 |

Confusion (rows = true, cols = predicted, order healthy, outer_race, inner_race, ball, cage):

```
    healthy    1776       0       1       0       0
 outer_race      19      91      14       0      95
 inner_race       9      22      67       0       0
       ball    1539       0     448       0       0
       cage      25      16       1       0       1
```

Per dataset (source-confounding check):

| Dataset | n | classes | macro-F1 | recall per class |
|---|---|---|---|---|
| ims | 2154 | 2 | 0.089 | healthy 1.00, ball 0.00 |
| xjtu_sy | 1970 | 4 | 0.565 | healthy 1.00, outer_race 0.42, inner_race 0.68, cage 0.02 |

Training-class sources (windows per class per dataset, train split):

```
{
 "cage": {
  "xjtu_sy": 406
 },
 "healthy": {
  "xjtu_sy": 3538
 },
 "inner_race": {
  "xjtu_sy": 67
 },
 "outer_race": {
  "xjtu_sy": 248
 }
}
```

## Detection

- Run-to-failure test faults caught by stage 3: 0.500 over 4 unit(s)
- Seeded-fault test records called correctly (3 consecutive windows): n/a over 0 record(s)

## Latency (single window, batch 1, 1 thread)

| Path | p50 | p95 | max |
|---|---|---|---|
| tft_torch | 58.15 ms | 62.92 ms | 66.54 ms |
| tft_onnx | 9.96 ms | 11.36 ms | 13.39 ms |
| nhits_torch | 11.67 ms | 13.08 ms | 15.70 ms |
| nhits_onnx | 0.42 ms | 0.79 ms | 2.23 ms |
| engine_step | 13.37 ms | 14.83 ms | 16.53 ms |

## Track: ims_finetuned

```
{
 "rul": {
  "S7_mape_pct": 468.5598570567735,
  "per_bearing_mape_pct": {
   "ims:test1:B4": 468.5598570567735
  },
  "mape_full_degradation_pct": 1093.738072311663,
  "per_bearing_mape_full_pct": {
   "ims:test1:B4": 1093.738072311663
  },
  "n_windows": 1730,
  "n_bearings": 1,
  "rmse_hours": 1101.3567128634877,
  "phm2012_score": 0.018542219260205293,
  "percent_of_life_error_pts": 108.49248554822253,
  "S7_mape_trend_only_pct": 1317.0457488226962,
  "S7_mape_onset_only_pct": 1484.901391686723,
  "per_dataset": {
   "ims": 468.5598570567735
  },
  "stage_within1_frac": 0.48840445269016697,
  "stage_mae": 1.592300556586271,
  "stage_max_error": 4,
  "stage_n": 2156
 },
 "classification": {
  "macro_f1_testable": 1.0,
  "testable_classes": [
   "healthy"
  ],
  "macro_f1_all5": 0.03566470902295782,
  "untestable": {
   "outer_race": "0 test support",
   "inner_race": "0 test support",
   "ball": "test support 1987 windows, 0 training support (no output channel)",
   "cage": "0 test support"
  },
  "per_dataset_recall": {
   "ims": {
    "healthy": 1.0,
    "ball": 0.0
   }
  },
  "note": "IMS B4 = ball only after onset; ball has no output channel, so the testable set is {healthy} alone and macro_f1_testable is NOT informative here; read macro_f1_all5/recall."
 },
 "detection": {
  "rtf_units": 1,
  "caught_by_stage3_frac": 0.0,
  "seeded_records": 0,
  "seeded_caught_frac": NaN
 }
}
```

## Track: ims_zero_shot_pretrained

```
{
 "rul": {
  "S7_mape_pct": 406.44963692064425,
  "per_bearing_mape_pct": {
   "ims:test1:B4": 406.44963692064425
  },
  "mape_full_degradation_pct": 710.2582186912636,
  "per_bearing_mape_full_pct": {
   "ims:test1:B4": 710.2582186912636
  },
  "n_windows": 1730,
  "n_bearings": 1,
  "rmse_hours": 1035.3297758951849,
  "phm2012_score": 0.020135645201889308,
  "percent_of_life_error_pts": 105.05581849875647,
  "S7_mape_trend_only_pct": 796.8855064378359,
  "S7_mape_onset_only_pct": 1196.474340810318,
  "per_dataset": {
   "ims": 406.44963692064425
  },
  "stage_within1_frac": 0.5426716141001855,
  "stage_mae": 1.4512987012987013,
  "stage_max_error": 3,
  "stage_n": 2156
 },
 "classification": {
  "macro_f1_testable": 1.0,
  "testable_classes": [
   "healthy"
  ],
  "macro_f1_all5": 0.08339575530586767,
  "untestable": {
   "outer_race": "0 test support",
   "inner_race": "0 test support",
   "ball": "test support 1987 windows, 0 training support (no output channel)",
   "cage": "0 test support"
  },
  "per_dataset_recall": {
   "ims": {
    "healthy": 1.0,
    "ball": 0.0
   }
  },
  "note": "IMS B4 = ball only after onset; ball has no output channel, so the testable set is {healthy} alone and macro_f1_testable is NOT informative here; read macro_f1_all5/recall."
 }
}
```

## Track: pretrain_test_only_rul

```
{
 "S7_mape_pct": 189.8770821269254,
 "per_bearing_mape_pct": {
  "xjtu_sy:Bearing1_4": 464.17882424224297,
  "xjtu_sy:Bearing2_5": 50.922442755116684,
  "xjtu_sy:Bearing3_4": 54.52997938341652
 }
}
```

## Track: pretrain_test_only_classification

```
{
 "macro_f1": 0.565301204796701,
 "n": 1970
}
```

## Track: SECONDARY_xjtu_lobo_robustness

```
{
 "file": "model_selection_robustness_lobo_20261004T214321Z.json",
 "note": "SECONDARY robustness table (LOBO over all eligible XJTU-SY run-to-failure bearings, incl. the frozen test bearings). Not the pre-registered S7 headline.",
 "mean_S7_mape_pct": 221.70976897739655,
 "median_S7_mape_pct": 73.70040964965521,
 "frac_bearings_meeting_15pct": 0.0,
 "per_bearing": {
  "xjtu_sy:Bearing1_1": {
   "S7_mape_pct": 96.79257295965705,
   "split": "train",
   "class": "outer_race",
   "stage_max_error": 2,
   "stage_within1_frac": 0.926829268292683
  },
  "xjtu_sy:Bearing1_2": {
   "S7_mape_pct": 70.96451545439184,
   "split": "train",
   "class": "outer_race",
   "stage_max_error": 2,
   "stage_within1_frac": 0.9254658385093167
  },
  "xjtu_sy:Bearing1_3": {
   "S7_mape_pct": 82.72509803841076,
   "split": "val",
   "class": "outer_race",
   "stage_max_error": 2,
   "stage_within1_frac": 0.9746835443037974
  },
  "xjtu_sy:Bearing1_4": {
   "S7_mape_pct": 569.8906349504728,
   "split": "test",
   "class": "cage",
   "stage_max_error": 4,
   "stage_within1_frac": 0.7786885245901639
  },
  "xjtu_sy:Bearing1_5": {
   "S7_mape_pct": 153.65251332648117,
   "split": "train",
   "class": "mixed",
   "stage_max_error": 3,
   "stage_within1_frac": 0.8461538461538461
  },
  "xjtu_sy:Bearing2_1": {
   "S7_mape_pct": 32.57592123501094,
   "split": "train",
   "class": "inner_race",
   "stage_max_error": 2,
   "stage_within1_frac": 0.9938900203665988
  },
  "xjtu_sy:Bearing2_2": {
   "S7_mape_pct": 73.70040964965521,
   "split": "val",
   "class": "outer_race",
   "stage_max_error": 3,
   "stage_within1_frac": 0.6459627329192547
  },
  "xjtu_sy:Bearing2_3": {
   "S7_mape_pct": 79.74074335463808,
   "split": "train",
   "class": "cage",
   "stage_max_error": 2,
   "stage_within1_frac": 0.9474671669793621
  },
  "xjtu_sy:Bearing2_4": {
   "S7_mape_pct": 27.108239016303198,
   "split": "train",
   "class": "outer_race",
   "stage_max_error": 5,
   "stage_within1_frac": 0.8095238095238095
  },
  "xjtu_sy:Bearing2_5": {
   "S7_mape_pct": 68.49399946315712,
   "split": "test",
   "class": "outer_race",
   "stage_max_error": 3,
   "stage_within1_frac": 0.8348082595870207
  },
  "xjtu_sy:Bearing3_1": {
   "S7_mape_pct": 46.69304462773424,
   "split": "train",
   "class": "outer_race",
   "stage_max_error": 2,
   "stage_within1_frac": 0.9980299448384555
  },
  "xjtu_sy:Bearing3_2": {
   "S7_mape_pct": 1174.4020011293787,
   "split": "train",
   "class": "mixed",
   "stage_max_error": 4,
   "stage_within1_frac": 0.8653846153846154
  },
  "xjtu_sy:Bearing3_3": {
   "S7_mape_pct": 72.63970354868185,
   "split": "train",
   "class": "inner_race",
   "stage_max_error": 2,
   "stage_within1_frac": 0.9730458221024259
  },
  "xjtu_sy:Bearing3_4": {
   "S7_mape_pct": 64.59191658690962,
   "split": "test",
   "class": "inner_race",
   "stage_max_error": 2,
   "stage_within1_frac": 0.9953795379537954
  },
  "xjtu_sy:Bearing3_5": {
   "S7_mape_pct": 711.6752213200659,
   "split": "val",
   "class": "outer_race",
   "stage_max_error": 4,
   "stage_within1_frac": 0.868421052631579
  }
 }
}
```

## Counts

```
{
 "ims": {
  "ft_test": {
   "units": 1,
   "windows": 2156
  },
  "ft_train": {
   "units": 3,
   "windows": 6468
  }
 },
 "xjtu_sy": {
  "test": {
   "units": 3,
   "windows": 1976
  },
  "train": {
   "units": 9,
   "windows": 6807
  },
  "val": {
   "units": 3,
   "windows": 433
  }
 }
}
```

Full JSON: `reports/spec_check_20261004T214921Z_real.json`
