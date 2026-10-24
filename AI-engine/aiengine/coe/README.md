# aiengine.coe — COE edge pipeline, hardware-ready

Python implementation of COE Components 1-9 (`Project architecture/COE/COE_Processing_Pipeline_Component_Details.md`),
fed through a `SampleSource` boundary so the AD7606/STM32 node can replace the simulator with no change downstream.

```
SampleSource ──256x8 int16 blocks + Keyphasor edges──> CoePipeline ──152-byte payload──> UDP :5005 ──> backend
 (sim | replay | hardware)        C1 window  C2 calibrate  C3 band-pass 2-8 kHz  C4 rectify
                                  C5 LP + /6 -> 5 kS/s   C6 Hann+4096 FFT   C7 32 features
                                  C8 1x amp/phase (2 probes)   C9 payload (aiengine/payload.py)
```

## Commands

```
cd AI-engine
python -m pytest tests/coe -q                                   # 23 tests
python -m aiengine.coe.run --source sim --fault outer_race --severity 0.5 --rpm 1750 --udp 127.0.0.1:5005 --rate 2
python -m aiengine.coe.run --source replay --file rec.npz --udp 127.0.0.1:5005
python -m aiengine.coe.verify [--quick]                         # writes reports/coe_verification.{json,md}
```
`record_blocks(source, "rec.npz", n)` records any source for replay. Use `.venv/Scripts/python.exe` once the venv is complete.

## Design notes

* Streaming exactly like the firmware: filters keep state across blocks, decimation phase continues, raw blocks are never concatenated;
  per window only the decimated envelope record (about 3.4 k floats/channel at 1750 rpm) and running sums are kept.
* Windows run Keyphasor edge to edge (20 revolutions); the sample at/after `t20` belongs to the next window; RPM = 1200 / (t20 - t0).
* Discarded (never emitted): lost DMA block (`seq` gap), speed outside 1750/3600 +-1 %, any revolution deviating >3 % from the window
  mean (also catches a missed Keyphasor edge), sample-count mismatch, non-finite result, stalled Keyphasor (timeout).
  After start-up or a lost block the filters must run 3000 sets (100 ms) before a window may begin.
* Features are plain sums / one rFFT / band sums: portable to C#. They differ slightly from `aiengine.dsp` (which uses zero-phase
  `sosfiltfilt` offline); this module is causal, as the STM32 is. Units: bp_rms in m/s2.

## Assumptions (COE doc ambiguous or silent) — confirm with COE

1. Envelope low-pass: 4th-order Butterworth at 2 kHz (= 5 kS/s / 2.5, same as `aiengine.dsp`); band-pass 4th-order Butterworth (8 poles).
2. Speed band +-1 % of 1750/3600 and +-3 % per-revolution deviation are placeholders for the commissioning values.
3. Below ~1465 rpm the 20-rev envelope record exceeds 4096 samples; the first 4096 are used and the window is flagged `truncated`.
   Only matters for the COE3 1000-3600 sweep; the two supported modes are not affected.
4. Proximity sensitivity default 7.87 mV/um (200 mV/mil class). RK4 value is not in the docs. Accelerometer 100 mV/g (spec S1), AD7606 +-10 V.
5. Displacement phase: x(theta) = A cos(theta - phi), theta = 0 at the Keyphasor edge.
6. `t20_ms` = t20 in ms since stream start (uint32).
7. Simulator amplitudes/resonance (5 kHz) are invented, `[SYNTHETIC]`; they are not rig-calibrated.
8. STM32 timing in `verify.py` assumes a 480 MHz Cortex-M7 (not stated in the COE doc).

## Hardware day: what to implement / swap

Preferred (the real design): the STM32 runs C1-C9 itself and sends the 152-byte payload (UART -> ESP32-C6 -> Wi-Fi -> UDP port 5005,
one payload per datagram, no framing). Then **nothing in this package is used at run time**: the backend already ingests it
(`UdpIngestService`, private/loopback senders only). Steps:

1. Firmware: implement C1-C9 per the COE doc; use this package as the golden reference.
2. Cross-check: record raw blocks from the ADC (below), run `CoePipeline(FileReplaySource(file))`, compare features to the firmware's
   payload for the same window. Feature order and payload layout: `aiengine/payload.py` (152 bytes asserted).
3. Commissioning values to enter in `CoeConfig` (and firmware): accelerometer sensitivity/offset per channel, proximity sensitivity,
   ADC range, speed band, per-revolution deviation, band-pass band if a kurtogram moves it.
4. Bring-up raw stream (optional): implement `HardwareSource.blocks()` for the frame format documented in its docstring
   (a proposal for the firmware team, not defined in the COE doc), or dump blocks to a file in the `.npz` layout of `record_blocks`
   (`codes` (N,256,8) int16, `seq` (N,), `edges_s` (M,)) and replay with `FileReplaySource`.
5. Re-measure on hardware and replace the `[CALCULATED]`/`[MEASURED host]` entries in `reports/coe_verification.md`: t20 to payload-ready on the
   STM32 (spec S5), sample count/overruns (S4), window loss and Wi-Fi loss (C6 assumptions).

Nothing downstream (payload codec, backend, models) changes: the boundary is the 152-byte payload.
