# rig_anchor plausibility: generator vs real windows

Provenance: [SIMULATION; endpoints resampled from rig windows, 1 bearing/fault, curve shape assumed].
Limits: No real run-to-failure data exists for 4 of 5 fault classes. Anchors = 5 physical bearings (H-01 + one per fault); test units are held-out simulated trajectories, not held-out bearings; the damage-curve family is assumed.

Cells are median [p5 - p95] of the raw node feature of channel B1Y (faulted housing, Y axis), in g (rms, envelope, bands) or unitless (kurtosis, crest).
Real = post-settle windows of the recorded run(s); generator = all splits, 1800 rpm and 3600 rpm pooled per class unless stated. 'gen anchor' = generator windows with late_w = 0 and g >= 0.98 (the measured fault level, by construction); 'gen end' = last 5 windows of a unit (u > 0.97).

## healthy
| set | n | rms | kurtosis | crest | envelope_rms | ftf | bsf | bpfo | bpfi |
|---|---|---|---|---|---|---|---|---|---|
| real healthy | 738 | 0.0304 [0.0245-0.04] | 4.31 [3.3-7.7] | 6.68 [4.66-9.07] | 0.025 [0.0205-0.0331] | 0.00169 [0.000744-0.00333] | 0.000925 [0.000413-0.00184] | 0.00114 [0.00056-0.00204] | 0.000957 [0.000429-0.00183] |
| gen (healthy units) | 34617 | 0.03 [0.0208-0.0495] | 4.31 [3.24-7.54] | 6.66 [4.57-9.21] | 0.0266 [0.0153-0.0394] | 0.00168 [0.000678-0.00355] | 0.000896 [0.000385-0.00187] | 0.00109 [0.000516-0.00217] | 0.000966 [0.000402-0.00213] |

## inner_race
| set | n | rms | kurtosis | crest | envelope_rms | ftf | bsf | bpfo | bpfi |
|---|---|---|---|---|---|---|---|---|---|
| real inner_race | 547 | 0.309 [0.246-0.363] | 3.65 [2.99-6.84] | 4.99 [3.97-8.26] | 0.261 [0.21-0.309] | 0.0155 [0.00708-0.0348] | 0.00959 [0.00481-0.0177] | 0.0147 [0.00713-0.0275] | 0.0165 [0.00635-0.0361] |
| gen run-in (u<0) | 9055 | 0.0324 [0.0189-0.0534] | 4.3 [3.24-7.53] | 6.73 [4.61-9.3] | 0.0275 [0.0181-0.0413] | 0.00181 [0.000738-0.00381] | 0.000931 [0.000398-0.00202] | 0.00116 [0.000527-0.00232] | 0.000973 [0.0004-0.00203] |
| gen anchor (g>=0.98, w=0) | 150 | 0.285 [0.17-0.459] | 3.19 [2.82-4.75] | 4.55 [3.79-6.72] | 0.243 [0.176-0.371] | 0.0139 [0.0063-0.0289] | 0.00862 [0.00436-0.0185] | 0.0127 [0.00685-0.0251] | 0.0139 [0.00651-0.0342] |
| gen end (u>0.97) | 561 | 0.822 [0.384-1.68] | 4.43 [3.65-7.75] | 6 [4.71-9.95] | 0.722 [0.406-1.57] | 0.057 [0.0176-0.179] | 0.0395 [0.0153-0.118] | 0.0469 [0.019-0.128] | 0.0421 [0.0161-0.138] |

## outer_race
| set | n | rms | kurtosis | crest | envelope_rms | ftf | bsf | bpfo | bpfi |
|---|---|---|---|---|---|---|---|---|---|
| real outer_race | 547 | 0.525 [0.472-0.556] | 3.52 [3.14-4.55] | 4.82 [4.05-5.98] | 0.437 [0.4-0.461] | 0.019 [0.01-0.0365] | 0.0139 [0.00694-0.0236] | 0.056 [0.0392-0.101] | 0.0157 [0.00824-0.0251] |
| gen run-in (u<0) | 8949 | 0.0301 [0.0192-0.0522] | 4.35 [3.24-7.63] | 6.69 [4.65-9.33] | 0.0262 [0.0156-0.0377] | 0.00161 [0.000669-0.00347] | 0.000899 [0.000371-0.00198] | 0.00109 [0.000506-0.0023] | 0.000957 [0.000398-0.00214] |
| gen anchor (g>=0.98, w=0) | 177 | 0.45 [0.353-0.743] | 4.07 [3.1-4.75] | 4.97 [4.19-5.85] | 0.403 [0.27-0.597] | 0.0176 [0.00919-0.0392] | 0.0125 [0.0067-0.0265] | 0.0708 [0.0352-0.136] | 0.0149 [0.00663-0.0249] |
| gen end (u>0.97) | 611 | 1.33 [0.782-3] | 5.53 [4.3-7.16] | 6.52 [5.36-8.14] | 1.23 [0.719-2.66] | 0.0815 [0.0298-0.216] | 0.0565 [0.0236-0.155] | 0.269 [0.105-0.789] | 0.0506 [0.0231-0.115] |

## ball
| set | n | rms | kurtosis | crest | envelope_rms | ftf | bsf | bpfo | bpfi |
|---|---|---|---|---|---|---|---|---|---|
| real ball | 181 | 0.15 [0.139-0.168] | 3.24 [3.03-4.68] | 4.61 [3.92-6.33] | 0.129 [0.119-0.144] | 0.00897 [0.00475-0.0189] | 0.00434 [0.00214-0.00767] | 0.00509 [0.00294-0.00864] | 0.00499 [0.00244-0.00935] |
| gen run-in (u<0) | 9108 | 0.0266 [0.0191-0.0384] | 5.85 [3.6-8.34] | 7.96 [5.87-9.63] | 0.0221 [0.0159-0.0304] | 0.00153 [0.000635-0.00354] | 0.000704 [0.000327-0.00139] | 0.00099 [0.00044-0.00189] | 0.000776 [0.00034-0.00159] |
| gen anchor (g>=0.98, w=0) | 151 | 0.154 [0.113-0.207] | 3.36 [2.96-5.02] | 4.78 [3.91-6.56] | 0.128 [0.0971-0.182] | 0.00947 [0.00452-0.0178] | 0.0045 [0.00207-0.00843] | 0.00489 [0.00246-0.00954] | 0.00524 [0.00239-0.011] |
| gen end (u>0.97) | 720 | 0.52 [0.278-0.943] | 4.49 [3.83-6.84] | 6.09 [4.81-8.71] | 0.456 [0.254-0.983] | 0.047 [0.0194-0.13] | 0.023 [0.00908-0.0724] | 0.0229 [0.0103-0.0607] | 0.0197 [0.00771-0.0557] |

## cage
| set | n | rms | kurtosis | crest | envelope_rms | ftf | bsf | bpfo | bpfi |
|---|---|---|---|---|---|---|---|---|---|
| real cage | 543 | 0.165 [0.143-0.225] | 4.05 [3.05-9.7] | 5.78 [4.1-8.43] | 0.14 [0.119-0.191] | 0.0116 [0.00374-0.0512] | 0.00506 [0.0027-0.00924] | 0.00631 [0.00294-0.0136] | 0.00628 [0.00331-0.0105] |
| gen run-in (u<0) | 10011 | 0.0316 [0.02-0.0484] | 4.32 [3.24-7.7] | 6.67 [4.59-9.15] | 0.0257 [0.0158-0.0422] | 0.00164 [0.00068-0.00358] | 0.000886 [0.000364-0.00193] | 0.00118 [0.00054-0.00249] | 0.000982 [0.000403-0.00199] |
| gen anchor (g>=0.98, w=0) | 118 | 0.163 [0.111-0.238] | 3.74 [2.95-8.9] | 5.36 [4.08-7.63] | 0.139 [0.0909-0.213] | 0.00791 [0.00304-0.0378] | 0.00489 [0.00258-0.00864] | 0.00695 [0.00306-0.017] | 0.006 [0.00264-0.0107] |
| gen end (u>0.97) | 560 | 0.576 [0.304-0.958] | 5.03 [3.87-12.3] | 6.78 [5.01-10.1] | 0.505 [0.281-0.859] | 0.0469 [0.0149-0.193] | 0.0264 [0.012-0.0608] | 0.0312 [0.0109-0.0887] | 0.0234 [0.0103-0.0559] |

## real ball failure trace (n=1; rpm column invalid; B1Y)
| segment | windows | rms | kurtosis | crest | envelope_rms | ftf | bsf | bpfo | bpfi |
|---|---|---|---|---|---|---|---|---|---|
| seg0 | 28 | 0.162 [0.15-0.183] | 3.22 [3.08-3.72] | 4.34 [3.85-5.35] | 0.134 [0.124-0.153] | 0.00863 [0.00451-0.0228] | 0.00392 [0.002-0.00736] | 0.00458 [0.00208-0.00902] | 0.00487 [0.00238-0.00891] |
| seg1 | 93 | 0.187 [0.0613-1.96] | 5.03 [2.82-129] | 6.31 [4.13-25.1] | 0.16 [0.0525-1.66] | 0.0108 [0.00088-0.225] | 0.00575 [0.000813-0.0792] | 0.00515 [0.000664-0.0571] | 0.00755 [0.00143-0.127] |
| seg2 | 23 | 0.575 [0.0213-1.64] | 4.27 [3.36-16.7] | 5.48 [4.25-10.5] | 0.504 [0.0164-1.44] | 0.0482 [0.00156-0.175] | 0.0235 [0.00101-0.0703] | 0.0222 [0.00083-0.0609] | 0.0195 [0.000672-0.0932] |

## Late amplification (measured on the real ball failure trace, B1X rms)
- measured rms(seg2 = windows 121-143) / rms(seg0 = windows 0-27) = **5.25** (seg1/seg0 = 1.61; envelope seg2/seg0 = 5.35) [MEASURED on rig, n=1 ball failure trace, rpm column invalid; segments 0-2 = windows 0-143]
- generator ball units: B1X rms at u>0.97 over the anchor level, median ratio = 4.96 (per-unit late scale lognormal sigma {'train': 0.2, 'val': 0.2, 'test': 0.35}; shape/anchor jitter blur it).

## Where the generator differs from real data (stated plainly)
- Anchor levels are the real fault-run levels BY CONSTRUCTION (resampled windows); agreement there is not a validation.
- Between healthy and anchor (g in (0,1)) the path is log-linear in every feature (assumed; no real mild/incipient data exists, real faults are already 5-19x healthy). Kurtosis/crest in that region interpolate between the real endpoints; in real windows healthy kurtosis is HIGHER than faulted (near-noise-floor signal), so the generator's kurtosis FALLS with damage until the late bump, the opposite of the textbook 'impulsive early' picture; unverified for the real rig.
- Beyond the anchor, behaviour comes from ONE failure trace (a ball bearing at 3600 rpm with an invalid rpm column) and is applied to all four fault classes; the late shape of inner/outer/cage failures is unknown.
- Window-to-window noise is real (block bootstrap of pool windows); with only 36-73 test-block windows per pool, test units re-use the same few real windows with different nuisance/curves, so test units are NOT independent trajectories.
- Per-unit mount/gain nuisance (sigma 0.20 / 0.10 log) is of the same order as the observed healthy run-to-run spread below (median smaller, max larger; only 2 runs).
- Time axis: 10-min snapshots with a life scale set by ISO 281 at an assumed duty (see rig_anchor_scale_check.md); the real rig windows are contiguous 0.67 s windows.

Observed log difference of per-run medians between the two healthy 1800 rpm runs (B1Y, 8 features): median 0.12, max 0.27 (n=2 runs, so only an indication) vs MOUNT_SIGMA 0.2.
