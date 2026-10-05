# AI-engine spec check (20261005T012112Z)

> **MEASURED on XJTU-SY / IMS / MaFaulDa public bearing datasets, not the team rig.** [MEASURED on xjtu_sy, ims, mafaulda] -- scored on held-out test bearings never used for training, early stopping, model selection or calibration. These are measurements of THOSE datasets' bearings; no rig data exists yet.

> **v2 = second look at XJTU/IMS test bearings; v2 choices made on validation only (reports/model_selection_tft_v2_20261005T011306Z.json). TFT trained jointly on XJTU-SY train + IMS ft_train + MaFaulDa train; N-HiTS + RUL calibration unchanged from v1 (run real).**

- Seed `20261004`, git `a4dd328` (+uncommitted changes)
- Command: `app.py evaluate --datasets xjtu_sy ims mafaulda --run real_v2 --tag real_v2 --robustness --threads 12 --note v2 = second look at XJTU/IMS test bearings; v2 choices made on validation only (reports/model_selection_tft_v2_20261005T011306Z.json). TFT trained jointly on XJTU-SY train + IMS ft_train + MaFaulDa train; N-HiTS + RUL calibration unchanged from v1 (run real).`
- Run dir: `real_v2`
- Host: Windows 10, Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, torch 2.14.0+cpu

## Verdicts

| Spec | Requirement | Target | Measured | Verdict |
|---|---|---|---|---|
| S7 | RUL MAPE on held-out degradation window (N-HiTS) | <= 15.0 % | 281.8 % | **FAIL** |
| S8 | TFT classification latency per window, p95 single-window | < 200.0 ms | 11.1 ms | **PASS** |
| IS3a | macro-F1 over the testable classes (train AND test support; primary) | >= 0.85  | 0.581 | **FAIL** |
| IS3a-5 | macro-F1 over all 5 classes (a class with no output channel scores F1 0) | >= 0.85  | 0.581 | **FAIL** |
| IS3b | stage error <= 1, literal reading: MAX abs stage error over test windows | <= 1  | 4.000 | **FAIL** |
| IS3c | run-to-failure faults caught (3 consecutive correct) by stage 3 | >= 1.0  | 0.250 | **FAIL** |
| IS1 (ICS leg) | features + TFT + N-HiTS + RUL per window, p95 | < 500.0 ms | 14.4 ms | **PASS** |

## S7 -- RUL detail (pre-registered metric, DESIGN.md section 7)

- S7 MAPE (bearing-averaged, degradation window minus last 10%): **281.8 %**
- MAPE over the full degradation window: 640.7 %
- RMSE: 1010.3 h; PHM-2012 score: 0.050; percent-of-life error: 94.2 pts
- Estimator components: trend-only 616.7 %, onset-only 1051.1 %
- Stage (IS3b): max abs error 4, mean abs error 0.901, fraction within +/-1 0.721 over 4132 windows

| Test bearing | MAPE (S7 window) | MAPE (full window) |
|---|---|---|
| ims:test1:B4 | 468.6 % | 1093.7 % |
| xjtu_sy:Bearing1_4 | 567.2 % | 1249.5 % |
| xjtu_sy:Bearing2_5 | 47.3 % | 153.1 % |
| xjtu_sy:Bearing3_4 | 44.1 % | 66.4 % |

## IS3 -- classification

- Test windows: 5569. **Primary IS3a macro-F1 over the testable classes ['healthy', 'outer_race', 'inner_race', 'ball', 'cage'] (5569 windows): 0.581**
- macro-F1 over all 5 classes: 0.581. Classes the model was trained on: ['healthy', 'outer_race', 'inner_race', 'ball', 'cage']. Not testable: {}
- Missed-fault rate 0.013, false-alarm rate 0.007

| Class | F1 | Support |
|---|---|---|
| healthy | 0.985 | 1909 |
| outer_race | 0.759 | 685 |
| inner_race | 0.017 | 98 |
| ball | 0.500 | 2352 |
| cage | 0.644 | 525 |

Confusion (rows = true, cols = predicted, order healthy, outer_race, inner_race, ball, cage):

```
    healthy    1896      13       0       0       0
 outer_race      13     541       5      51      75
 inner_race      11      17      14      47       9
       ball      11      13    1496     831       1
       cage      11     156      27      41     290
```

Per dataset (source-confounding check):

| Dataset | n | classes | macro-F1 | recall per class |
|---|---|---|---|---|
| ims | 2154 | 2 | 0.679 | healthy 1.00, ball 0.24 |
| mafaulda | 1445 | 4 | 0.848 | healthy 0.90, outer_race 0.93, ball 0.96, cage 0.60 |
| xjtu_sy | 1970 | 4 | 0.450 | healthy 1.00, outer_race 0.49, inner_race 0.14, cage 0.00 |

Confusion, ims only (rows = true, cols = predicted, order healthy, outer_race, inner_race, ball, cage):

```
    healthy     167       0       0       0       0
 outer_race       0       0       0       0       0
 inner_race       0       0       0       0       0
       ball      11       0    1496     480       0
       cage       0       0       0       0       0
```

Confusion, mafaulda only (rows = true, cols = predicted, order healthy, outer_race, inner_race, ball, cage):

```
    healthy     119      13       0       0       0
 outer_race       2     434       0      13      17
 inner_race       0       0       0       0       0
       ball       0      13       0     351       1
       cage       0     151       0      41     290
```

Confusion, xjtu_sy only (rows = true, cols = predicted, order healthy, outer_race, inner_race, ball, cage):

```
    healthy    1610       0       0       0       0
 outer_race      11     107       5      38      58
 inner_race      11      17      14      47       9
       ball       0       0       0       0       0
       cage      11       5      27       0       0
```

Training-class sources (windows per class per dataset, train split):

```
{
 "ball": {
  "mafaulda": 1636
 },
 "cage": {
  "mafaulda": 1896,
  "xjtu_sy": 406
 },
 "healthy": {
  "ims": 4482,
  "mafaulda": 516,
  "xjtu_sy": 3538
 },
 "inner_race": {
  "ims": 1986,
  "xjtu_sy": 67
 },
 "outer_race": {
  "mafaulda": 1874,
  "xjtu_sy": 248
 }
}
```

## Detection

- Run-to-failure test faults caught by stage 3: 0.250 over 4 unit(s)
- Seeded-fault test records called correctly (3 consecutive windows): 0.606 over 213 record(s)

## Latency (single window, batch 1, 1 thread)

| Path | p50 | p95 | max |
|---|---|---|---|
| tft_torch | 57.89 ms | 63.34 ms | 68.34 ms |
| tft_onnx | 9.72 ms | 11.14 ms | 15.51 ms |
| nhits_torch | 11.88 ms | 13.47 ms | 14.78 ms |
| nhits_onnx | 0.42 ms | 0.61 ms | 1.30 ms |
| engine_step | 12.96 ms | 14.42 ms | 18.90 ms |

## Track: ims_finetuned

```
{
 "rul": {
  "S7_mape_pct": 468.55985705990497,
  "per_bearing_mape_pct": {
   "ims:test1:B4": 468.55985705990497
  },
  "mape_full_degradation_pct": 1093.738072314391,
  "per_bearing_mape_full_pct": {
   "ims:test1:B4": 1093.738072314391
  },
  "n_windows": 1730,
  "n_bearings": 1,
  "rmse_hours": 1101.3567128422028,
  "phm2012_score": 0.018542219264628335,
  "percent_of_life_error_pts": 108.4924855377795,
  "S7_mape_trend_only_pct": 1317.0457496718318,
  "S7_mape_onset_only_pct": 1484.9013911673983,
  "per_dataset": {
   "ims": 468.55985705990497
  },
  "stage_within1_frac": 0.48840445269016697,
  "stage_mae": 1.592300556586271,
  "stage_max_error": 4,
  "stage_n": 2156
 },
 "classification": {
  "macro_f1_testable": 0.6786262725953602,
  "testable_classes": [
   "healthy",
   "ball"
  ],
  "macro_f1_all5": 0.2714505090381441,
  "untestable": {
   "outer_race": "0 test support",
   "inner_race": "0 test support",
   "cage": "0 test support"
  },
  "per_dataset_recall": {
   "ims": {
    "healthy": 1.0,
    "ball": 0.24157020634121792
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
  "S7_mape_pct": 577.1252574626135,
  "per_bearing_mape_pct": {
   "ims:test1:B4": 577.1252574626135
  },
  "mape_full_degradation_pct": 985.0573748547741,
  "per_bearing_mape_full_pct": {
   "ims:test1:B4": 985.0573748547741
  },
  "n_windows": 1730,
  "n_bearings": 1,
  "rmse_hours": 1417.533672355877,
  "phm2012_score": 0.020535393686228653,
  "percent_of_life_error_pts": 141.3564384431936,
  "S7_mape_trend_only_pct": 796.469744232218,
  "S7_mape_onset_only_pct": 1467.775915508208,
  "per_dataset": {
   "ims": 577.1252574626135
  },
  "stage_within1_frac": 0.48237476808905383,
  "stage_mae": 1.6781076066790352,
  "stage_max_error": 4,
  "stage_n": 2156
 },
 "classification": {
  "macro_f1_testable": 0.6786262725953602,
  "testable_classes": [
   "healthy",
   "ball"
  ],
  "macro_f1_all5": 0.2714505090381441,
  "untestable": {
   "outer_race": "0 test support",
   "inner_race": "0 test support",
   "cage": "0 test support"
  },
  "per_dataset_recall": {
   "ims": {
    "healthy": 1.0,
    "ball": 0.24157020634121792
   }
  },
  "note": "IMS B4 = ball only after onset; ball has no output channel, so the testable set is {healthy} alone and macro_f1_testable is NOT informative here; read macro_f1_all5/recall."
 }
}
```

## Track: pretrain_test_only_rul

```
{
 "S7_mape_pct": 219.5539491513243,
 "per_bearing_mape_pct": {
  "xjtu_sy:Bearing1_4": 567.2088998349585,
  "xjtu_sy:Bearing2_5": 47.31991206631444,
  "xjtu_sy:Bearing3_4": 44.13303555269989
 }
}
```

## Track: pretrain_test_only_classification

```
{
 "macro_f1": 0.6811097900674866,
 "n": 3415
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
 "mafaulda": {
  "test": {
   "units": 233,
   "windows": 1911
  },
  "train": {
   "units": 701,
   "windows": 5922
  },
  "val": {
   "units": 235,
   "windows": 1907
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

Full JSON: `reports/spec_check_20261005T012112Z_real_v2.json`
