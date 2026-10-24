# COE pipeline verification

Generated 2026-10-24 04:21:36 | commit 042bcf4 | seed 20261024 | command `python -m aiengine.coe.verify`

ALL DATA SIMULATED. Tags: [SIMULATION] code run on simulated samples; [MEASURED host] wall time on this PC (not STM32); [CALCULATED] arithmetic from the COE doc / stated assumptions. Nothing is a rig measurement.

| Metric | Value | Verdict | Provenance |
|---|---|---|---|
| COE1 sample rate x channels | 30.0 kS/s x 4 ch (target >=25); 9.993 s simulated, 299776 sets/ch, gap-free=True; ADC conv+read 17.0 us of 33.333 us | MET (design) [CALCULATED + SIMULATION]; hardware test pending | [CALCULATED]+[SIMULATION] |
| COE2 host t20->payload @1750 rpm (245 windows) | p50 5.34 / p95 6.81 / max 8.98 ms (finalize only p50 3.65, max 5.82) | <=333 ms on host PC | [MEASURED host] |
| COE2 host t20->payload @3600 rpm (245 windows) | p50 5.10 / p95 6.25 / max 7.80 ms (finalize only p50 3.57, max 5.90) | <=333 ms on host PC | [MEASURED host] |
| COE2 STM32H755 estimate @1750 rpm | 955202 flops after t20; 12.51 ms nominal / 28.43 ms pessimistic incl. 8.533 ms block wait | <=333 ms (margin 11.7x pessimistic) | [CALCULATED] |
| COE2 STM32H755 estimate @3600 rpm | 775488 flops after t20; 11.76 ms nominal / 24.69 ms pessimistic incl. 8.533 ms block wait | <=333 ms (margin 13.5x pessimistic) | [CALCULATED] |
| COE3 fixed dimension | 28 speeds 1000-3600: payload bytes [152], feature shapes [[32]], BPFO dominant at 28/28 | MET (design) [SIMULATION] | [SIMULATION] |
| C6 window loss, 1.0 h @ 1750 rpm | node 0.038 % / end-to-end 0.133 % (limit 1 %); valid 5247, discarded 3, Wi-Fi lost 5 of 5249 ideal | MET under stated loss assumptions [SIMULATION] | [SIMULATION] |

## Caveats

- COE2: wall time of the Python/NumPy finalisation on this PC, NOT STM32 time. finalize_ms = Components 6-9; t20_to_payload_ms = processing of the block holding t20 (filters+stats for that 256-set block) + finalize. Host: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel / Windows-10-10.0.26300-SP0.
- COE3: Below ~1465 rpm a 20-rev envelope record exceeds 4096 samples (1200/rpm*5000); the COE doc fixes the FFT at 4096 so the first 4096 samples are used (flag 'envelope_record_truncated_to_4096'). The doc's own supported modes are 1750 and 3600 rpm, where nothing is truncated. Open question for COE.
- C6: ASSUMED rates, not measured on the rig or on the team's Wi-Fi; replace with measured values when hardware exists. Loss model: block drop 1e-05, edge miss 1e-05; Gilbert-Elliott burst model: P(good->bad)=0.001 per payload, P(bad->good)=0.3, loss prob in bad state 0.8, 0 in good; no retransmission (COE doc: no acks/retries). Stationary mean loss ~0.27 %.
- C6: Window loss = (ideal back-to-back 20-rev windows in the period - windows delivered) / ideal. A lost block also costs the filter re-settle (100 ms) before the next window may start. Spec-sheet C6 is currently the ME enclosure constraint; the wireless-reliability constraint is only a proposal.

## STM32 estimate assumptions

- Cortex-M7 at 480 MHz (ST datasheet maximum; not stated in the COE doc, confirm)
- 5*N*log2(N) flops for the 2048-pt complex FFT behind each 4096-pt real FFT
- ~70 flops per sample-channel for calibrate + 4 BP biquads + rectify + 2 LP biquads + moment sums
- ~150 flops per proximity sample per probe for the last revolution (sin/cos totals)
- 2 cycles/flop nominal, 10 cycles/flop pessimistic; ADC wait for the DMA block holding t20 added (8.533 ms)
- UART transfer to the ESP32-C6 is outside the 'payload available in STM32 memory' definition
