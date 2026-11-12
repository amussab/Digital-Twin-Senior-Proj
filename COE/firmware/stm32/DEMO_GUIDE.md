# Demo guide: showing the specifications and constraints

Follow the parts in order. About 20 minutes. Details and background: `README.md` section 1.

| Part | Shows |
| --- | --- |
| A | S2 processing time and S3 fixed features, on recorded data |
| B | S1 sampling rate, 4 channels |
| C | S1 + S2 + S3 live: ADC → DSP → 32 features |
| D | C1 all processing local |
| E | C2 sensor power (24 V, about 4 mA) |
| F (extra) | The low-pass (anti-aliasing) filter's effect |

---

## 0. Before the coach arrives (10 minutes)

**Wiring check (24 V supply OFF):**

1. STM32 USB cable plugged into the PC.
2. ADC wires as in `README.md` 3.1: RST → CN7-2, BUSY → CN7-4, GND → CN7-8,
   RD → CN7-10, D7 → CN7-12, CS → CN7-16, CA → CN7-18, D15 → CN7-20,
   OS0 → CN7-1, OS1 → CN7-3, OS2 → CN7-7, RANGE → CN7-11, CB → CN10-16.
3. ADC module 5V and GND wired **straight** from the 24 → 5 V converter.
4. Hi-Link connected to the 24 V (gives the filter its ±15 V).
5. **Generator → filter IN1** (signal), generator ground → common 0 V.
6. **Filter OUT1 → AIN2, AIN4, AIN6, AIN8** (all four joined). Filter OUT2, OUT3,
   OUT4 connected to **nothing**.

**Generator (FNIRSI 2C53T):** Signal Generator screen → **Sine Wave**,
Frequency **5000 Hz**, Amplitude **1.0 V**, output **on** (centre ▷‖ button).

**Two PowerShell windows**, both in the firmware folder. In each, type:

```powershell
cd "C:\Users\USER2\OneDrive - KFUPM\Senior Design\coe_firmware"
```

- **Window 1** is for commands.
- **Window 2** shows the board's messages. Start it now and leave it open the whole demo:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\com.ps1
```

Close VS Code debug sessions (the scripts use the same ST-LINK).

**Reminders**
- **B1** = the **blue** button on the STM32 board (not the black RESET button).
- Power order: USB on → **24 V on** → B1. Stopping: **B1** → 24 V off.
- Green LED: slow blink = waiting, fast blink = acquisition running. Red LED = an error counter is not 0.

---

## A. S2 and S3 on recorded data

*Say: "First the processing on recorded test data, which you approved before: the
STM32 checks itself against an independent Python reference."*

1. 24 V can stay **off** (the ADC is not used here).
2. Window 1:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\tools\flash.ps1 -Mode REPLAY
   ```

   Wait for `Flashed REPLAY on both cores and reset the board.` (about 1 minute).
3. Window 1:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\tools\replay_results.ps1
   ```

   It waits 8 s, then prints 39 lines.
4. **Point at:**
   - Last line: `All replay checks passed.`
   - **S2:** `pipeline_batch_max_ms_1750  91.03` and `pipeline_batch_max_ms_3600  47.87`
     (both ≤ 333 ms), `pipeline_batch_deadline_met  True`.
   - **S3:** `pipeline_batch_fixed_shape_passed  True` (32 features at both speeds),
     `pipeline_batch_checks_passed  True` (all 32 values match Python).
   - `SystemCoreClock 400000000`, `processing_clock_source 1` (crystal-referenced clock).

---

## B. S1: sampling at ≥ 25,000 samples/s on 4 channels

*Say: "The AD7606 samples all four channels together. The STM32 measures the real
rate from the ADC's own BUSY signal every second and counts any lost sample."*

1. 24 V **off**. Window 1:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\tools\flash.ps1 -Mode LIVE_RAW
   ```

2. Window 2 shows `=== COE CM7 LIVE_RAW: AD7606 raw acquisition ===`.
3. Switch the **24 V on**, wait 3 s, press **B1**.
4. Window 2 shows `AD7606 started: 29998.50 S/s per channel ...`, then one report per second, like:

   ```text
   [60s] rate 29998.50 Hz (avg 29998.50) frames 1799895 | missed 0 overlap 0 overrun 0 dma 0 spi 0 stall 0
     V2 AIN2 B1X    512.90    333.73    947.11     255..6462       0
     V4 AIN4 B1Y    513.06    333.76    947.42     255..6464       0
     V6 AIN6 B2X    512.84    333.77    947.27     254..6462       0
     V8 AIN8 B2Y    513.17    333.78    946.66     260..6464       0
   ```

5. **Point at:**
   - `rate 29998.50 Hz`: **per channel**, above 25,000. It is measured, not just set.
   - `missed 0 overlap 0 overrun 0 dma 0 spi 0`: **no sample lost**, every second.
   - The four channels **V2, V4, V6, V8** (B1X, B1Y, B2X, B2Y) all read the same
     5-kHz signal (about 334 mV RMS): all four inputs are sampled correctly.
     (V1, V3, V5, V7 are unused inputs.)
6. Optional: change the generator to 1 kHz or 8 kHz; the numbers change on all four together.
7. After 30–60 s press **B1** (stop). Leave the 24 V on for part C.

---

## C. S1 + S2 + S3 live: the full chain

*Say: "Now the real system: samples from the ADC go straight into the DSP on the
STM32, and every window produces the 32 features."*

1. Press **B1** only if acquisition is still running (to stop it). Window 1:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\tools\flash.ps1 -Mode LIVE
   ```

   (Flashing resets the board. Switch the 24 V off first if you want to be careful.)
2. Window 2 shows `=== COE CM7 LIVE: AD7606 -> DSP -> ESP32 ===`.
3. **24 V on**, press **B1**. Every 0.7 s a window appears, like:

   ```text
     window 7 (20571 samples/channel, 1750 RPM): features ready 10.83 ms after its last sample (max 11.11 ms, limit 333) OK
       32 features (amplitudes in mV; kurtosis and crest factor plain), valid mask 0xf:
       B1X: RMS 334.44 kurt 1.50 crest 1.42 envRMS 301.38 FTF 0.05 BSF 0.07 BPFO 0.10 BPFI 0.43
       B1Y: RMS 334.46 kurt 1.50 crest 1.42 envRMS 301.41 FTF 0.05 BSF 0.07 BPFO 0.10 BPFI 0.43
       B2X: RMS 334.47 kurt 1.50 crest 1.42 envRMS 301.41 FTF 0.05 BSF 0.06 BPFO 0.10 BPFI 0.43
       B2Y: RMS 334.48 kurt 1.50 crest 1.42 envRMS 301.42 FTF 0.05 BSF 0.07 BPFO 0.10 BPFI 0.43
   ```

   and once per second:

   ```text
   [5s] rate 29998.50 Hz | windows 7 ... | gaps 0 dropped blocks 0 restarts 0 stall 0
     ... window latency max 11.11 ms (limit 333) ...
   ```

4. **Point at:**
   - **S2:** `features ready 10.83 ms after its last sample ... limit 333 OK`. Every
     window is about 11 ms (worst seen 16 ms), about 20× inside the limit.
   - **S3:** exactly **8 features × 4 channels = 32**, always in this order
     (RMS, kurtosis, crest, envelope RMS, FTF, BSF, BPFO, BPFI), every window;
     `valid mask 0xf` = all four channels valid.
   - **The values are correct:** for a sine wave theory gives kurtosis **1.5**,
     crest factor **√2 = 1.41**, RMS = peak / √2 (**about 334 mV** here).
   - **S1 still holds:** `rate 29998.50 Hz`, `gaps 0 dropped blocks 0`.
   - `packet not sent (uncalibrated)` is on purpose: a generator voltage is not
     acceleration, so it is not sent as g. (With the real sensor, 100 mV/g is set
     in `CM7/App/config/live_config.h` and packets go to the ESP32; the ESP32
     delivery was shown with recorded data.) Since 2 Nov 2026 the calibration
     is set (0.100 V/g); to repeat this generator demo, set
     `COE_LIVE_VOLTS_PER_G` back to all 0 first.
5. After about 30 s press **B1**, then switch the **24 V off**.

---

## D. C1: all processing happens locally

*Say: "Everything you just saw was computed on the STM32 itself."*

1. Point out that in part C the **ESP32 was not needed** and **no PC software
   computed anything**: Window 2 only displays text that the STM32 prints.
2. The DSP code runs on the STM32's Cortex-M7: folders `CM7/App/dsp` (filters,
   envelope, FFT, features) and `CM7/App/pipeline`.
3. The ESP32 only receives the finished 160-byte packet and forwards it.

---

## E. C2: sensor power, 24 V and about 4 mA

*Say: "The conditioner powers the IEPE sensor from the 24-V supply with a constant
current of about 4 mA."*

**24 V**

1. Show the bench supply display: about **24 V**.
2. Optional: multimeter on DC V, black probe on the conditioner GND, red probe on
   its rear **+24V IN**: about **24 V**.

**4 mA** (same method as recorded on 23 Oct, about 4.3 mA)

1. **24 V off.** Unplug the sensor from the conditioner's front **IEPE 0** socket.
2. Multimeter: black lead in **COM**, red lead moved to **mA**, dial on **DC mA**.
3. Use the BNC-to-clips cable on the **front IEPE 0** socket and connect one chain:
   **BNC red clip → meter red probe → meter → meter black probe → 3.3 kΩ resistor → BNC black clip.**
4. **24 V on.** Read the meter: about **4.3 mA** (recorded: 4.275 mA).
5. **24 V off.** Move the red meter lead back to **V/Ω**. Plug the sensor back in.

---

## F. Extra: the low-pass filter (not a specification)

*Say: "The analog filter is a 2nd-order unity-gain low-pass at 11.5 kHz. It lets
the 2–8 kHz band through and reduces higher frequencies before the ADC, so they
cannot fold back into our band (the ADC sampling at 30 kS/s can only represent
up to 15 kHz)."*

1. Wiring as in part 0 (generator → filter IN1, OUT1 → AIN2/4/6/8). Generator:
   sine, **1.0 V**, start at **1000 Hz**.
2. 24 V **off**. Window 1:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\tools\flash.ps1 -Mode LIVE_RAW
   ```

3. **24 V on**, press **B1**. Watch the line `V2 AIN2 B1X` in Window 2; the second
   number after the name is the **RMS in mV**.
4. Change the generator frequency, wait 2–3 s at each, and read the V2 RMS:

   | Generator | Expected V2 RMS | Passes | What to say |
   | ---: | ---: | ---: | --- |
   | 1 kHz | about **351 mV** | 100 % | reference: fully passed |
   | 5 kHz | about **334 mV** | 95 % | middle of the DSP band: passed |
   | 8 kHz | about **293 mV** | 83 % | top of the band: still passed |
   | 14 kHz | about **154 mV** | 44 % | above the cutoff: clearly reduced |
   | 20 kHz | about **70 mV** (predicted) | about 20 % | above 15 kHz: strongly reduced; it appears as 10 kHz (aliasing), which is why the filter is needed |

   1–14 kHz are measured values (26 Oct 2026). 20 kHz is predicted from the
   fitted filter model: show it live and note the number.
   (Full scale: a 1 V peak-to-peak sine is 354 mV RMS with no loss.)
5. **Show the full measured curve:** open
   `CM7/TestData/results/live/filter_response.png` (gain vs frequency, −3 dB at
   10 kHz for filter + ADC) and the table in `README.md` section 4.7.
6. If asked why there is loss below 11.5 kHz: a 2nd-order filter is already
   −3 dB *at* its cutoff and rolls off gradually below it, and the AD7606 has
   its own built-in anti-alias filter. Fitting both to the measurement gives the
   board filter **11.6 kHz, Q 0.75**, matching the 11.5-kHz Butterworth design.
7. Press **B1** to stop, then 24 V off.

How the curve was measured (if asked): while LIVE_RAW runs,
`tools\ring_snapshots.ps1` saves the raw samples every 2 s, and
`tools\ring_sine_fit.py` fits the frequency and amplitude in each (README 4.7).

---

## Finish

- Press **B1** if acquisition runs, then **24 V off**, then unplug USB.
- Window 2: **Ctrl+C**.
- Evidence files to show if asked: `CM7/TestData/results/live/live_5khz_run.txt`
  (a full live run) and `CM7/TestData/results/live/filter_response.png`.

## If something goes wrong

| What you see | What to do |
| --- | --- |
| `AD7606 start failed: no BUSY pulse ...` | The 24 V is off, or the ADC 5 V wire is loose. Switch the 24 V on, check, press B1 again |
| `AD7606 start failed: BUSY stayed high ...` | Check that CA (CN7-18) and CB (CN10-16) are both connected, then press B1 again |
| Window 2: `Access to the port 'COM5' is denied` | Another program uses the port: close other terminals or serial monitors, start Window 2 again |
| `Flashing failed` | Close any VS Code debug session; check the USB cable; run the command again |
| Window 2 shows nothing after B1 | Press the **blue** B1 (not black RESET); check Window 2 is still running |
| The four channels show different values | A wire from filter OUT1 to an AIN input is loose |
| A replay check shows FAIL | Run `replay_results.ps1` again after 10 s (checks may still be running) |
