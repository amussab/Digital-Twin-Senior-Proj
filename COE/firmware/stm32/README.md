# COE firmware: STM32H755 acquisition and DSP layer

**Showing the specifications to someone? Follow [DEMO_GUIDE.md](DEMO_GUIDE.md).**

This is the one guide for the STM32 firmware. Detailed history lives in git
(`git log`); everything you need to build, wire, run and check the board is here.

```text
sensor -> IEPE conditioner -> filter -> AD7606 ADC (4 of 8 inputs)
       -> STM32H755 CM7: 30 kS/s per channel -> DSP -> 32 features per window
       -> 160-byte packet -> USART2 -> ESP32-C6 -> Wi-Fi/HTTP -> host
```

The Cortex-M7 does all the work. The Cortex-M4 only starts and idles.
Sibling projects: `../coe_esp32` (receiver/forwarder, ESP-IDF) and
`../coe_pipeline` (Python reference for every DSP step).

## Contents

1. [Specifications, evidence and demo](#1-specifications-evidence-and-demo)
2. [Build, flash and test (commands)](#2-build-flash-and-test)
3. [Wiring and power](#3-wiring-and-power)
4. [AD7606 acquisition](#4-ad7606-acquisition)
5. [DSP pipeline](#5-dsp-pipeline)
6. [Feature vector and packet](#6-feature-vector-and-packet)
7. [STM32 to ESP32 link](#7-stm32-to-esp32-link)
8. [Verification and Watch variables](#8-verification-and-watch-variables)
9. [Clock and memory](#9-clock-and-memory)
10. [Project layout and build notes](#10-project-layout-and-build-notes)
11. [Open items](#11-open-items)

## 1. Specifications, evidence and demo

### 1.1 Evidence (measured on the board)

| Item | Requirement | Result | Evidence |
| --- | --- | --- | --- |
| **S1** Sampling | 4 channels, ≥ 25,000 S/s each | ✅ **29,998.50 S/s per channel**, measured from BUSY timestamps every second; **0 missed / lost samples** (45 s, 133 s and 44 s runs); the four inputs agree within about 1 mV on the same signal | 4.5, 4.7; `CM7/TestData/results/live/live_5khz_run.txt` |
| **S2** Processing time | Features within 333 ms of a window's end | ✅ Replay (approved): **91.0 ms** (1750 RPM) / **47.9 ms** (3600 RPM). **Live** with the ADC: **64 of 64 windows OK**, typically 10–11 ms, **worst 16.1 ms** after the window's last sample; DSP load 16% | 8; `CM7/TestData/results/live/live_5khz_run.txt` |
| **S3** Fixed AI features | Same 32 features in the same order every window | ✅ Replay (approved): 32 at both speeds, all checks against Python. **Live**: every window prints exactly 32 values, valid mask 0xf, at 1750 RPM (20,571 samples/channel). With a 5-kHz sine the values match theory: kurtosis **1.50** (1.5), crest **1.42** (√2), RMS **334.4 mV** (0.944 V pp ÷ 2√2 = 333.8 mV) | live run log |
| **C1** Local processing | All processing on the node | ✅ All DSP and feature extraction run on the STM32 Cortex-M7 (`CM7/App`); the live run above used no network (ESP32 not even connected). The ESP32 only forwards finished packets | 5, 7 |
| **C2** Sensor power | IEPE excitation from the 24-V supply, about 4 mA | ✅ 24 V measured at the conditioner; IEPE 0 (the channel in use) **4.275 mA** (cross-check 4.287 mA). IEPE 1–3 not yet measured | 3.4 |

Related measurements: filter + ADC response −3.0 dB at **10 kHz**, −1.6 dB at
8 kHz (4.7, plot `CM7/TestData/results/live/filter_response.png`); replay feature
delivery STM32 → ESP32 → host (7). Independent rate check: generator tones of
1–14 kHz were measured within 0.6 % of their set values (generator setting
accuracy), consistent with the 29,998.5-S/s sample clock.

### 1.2 Demo script (about 15 minutes)

Wiring as in section 3, generator (FNIRSI 2C53T, sine, 1 V) → filter IN1,
filter OUT1 → AIN2/4/6/8, generator ground to the common 0 V. Commands run in
PowerShell from `coe_firmware` (section 2.2).

1. **S2 and S3 on recorded data (approved before).** 24 V off.
   `powershell -ExecutionPolicy Bypass -File .\tools\flash.ps1 -Mode REPLAY`, then
   `powershell -ExecutionPolicy Bypass -File .\tools\replay_results.ps1`:
   39 checks PASS, 91/48 ms, 32 features. With the ESP32 and host receiver
   running (2.6), its monitor shows the 12 received packets.
2. **S1 live.** `.\tools\flash.ps1 -Mode LIVE_RAW`, open `.\tools\com.ps1`,
   generator 5 kHz, 24 V on, **B1**: each second shows `rate 29998.50 Hz` and
   `missed 0 overlap 0 overrun 0`; the four slots V2/V4/V6/V8 show the same
   signal. B1 to stop.
3. **S1 + S2 + S3 live, the full chain.** `.\tools\flash.ps1 -Mode LIVE`,
   `.\tools\com.ps1`, 24 V on, **B1**: about every 0.7 s a line
   `window N ... features ready ~11 ms after its last sample ... limit 333 OK`
   followed by the 32 features. Point out kurtosis 1.50 and crest 1.42 for the
   sine. B1, then 24 V off.
4. **C1.** Same run as step 3: no network or PC computation is involved; the
   STM32 prints the finished features itself.
5. **C2.** Multimeter in series with the IEPE 0 input as in 3.4 (about 4.3 mA),
   and the 24 V reading on the bench supply.
6. Optional: the frequency-response plot (4.7) and the filter-sweep table.

Sensor note: since 2 Nov 2026 the sensors are **TE Model 805-0050** IEPE
accelerometers (`../sensor_datasheet.pdf`), nominal **100 mV/g** (±10 %),
±50 g, flat ±1 dB to 8 kHz. Pin 1 = +OUT, pin 2 = GND (case), pin 3 unused.
`COE_LIVE_VOLTS_PER_G` is now `{0.100f, ...}` (conditioner gain assumed 1,
filter gain 1.000 measured at 100 Hz), so LIVE sends packets. For the
generator demo (part F of the demo guide) set it back to 0, because a
generator signal is not acceleration. The earlier sensor was a PCB 603C01
(SN 32495, also 100 mV/g).

### 1.3 Status of the parts

| Part | State (26 Oct 2026) |
| --- | --- |
| DSP pipeline, 32 features, packet | Done; every stage checked on the board against Python references |
| UART to ESP32, delivery timing | Done; see section 7 |
| Machine test, 250 RPM | **Working (11 Nov 2026)**: 318 windows, all received by the laptop, 0 gaps, 0 errors, max 16.3 ms. Machine off about 1.2 mg (kurtosis about 3); running B1X about 7.3, B1Y about 4.6, B2Y about 6.8 mg (kurtosis about 3.3) |
| Full chain to the host | **Working (11 Nov 2026)**: sensors → STM32 (LIVE) → ESP32 → phone hotspot → laptop; every packet accepted, 0 gaps, 0 UART errors (2.6) |
| Clock | 400 MHz from the ST-LINK 8 MHz (HSE); replay checks re-passed on it |
| AD7606 acquisition (LIVE_RAW) | Working, 4 channels, 29,998.50 S/s, 0 lost (4.5) |
| Live chain (LIVE) | **Working with the real sensors (2 Nov 2026)**: 156/156 windows sent, 0 gaps, worst latency 16.1 ms, features in g (4.8) |
| Analog filter + ADC | Measured: −3.0 dB at 10 kHz (4.7) |
| Sensor signal | **Working (2 Nov 2026)**: three TE 805-0050 sensors, channel map confirmed by tap test, quiet noise 0.3 mV RMS (about 3 mg) and offsets under 7 mV on all four channels (4.8). 11 Nov 2026, final cabling: still noise about 4.3 mg (STM32 on USB) or about 12.5 mg (STM32 on the converter), same on all channels; accepted as the noise floor (3.4) |
| AD7606 module | Works. The earlier "fault" was the CB wire on CN7 pin 17 instead of 18 (4.5) |
| Calibration (volts per g) | Set 2 Nov 2026: 0.100 V/g on all channels (TE 805-0050 nominal); not yet checked against a known vibration |

## 2. Build, flash and test

### 2.1 Firmware modes

The firmware has three modes. **[CM7/firmware_mode.cmake](CM7/firmware_mode.cmake)**
holds one line that picks the mode for VS Code builds; the scripts below take
the mode as an option instead.

| Mode | What it does | Use it to |
| --- | --- | --- |
| `REPLAY` (default) | Runs all recorded-data DSP checks at start-up, sends 12 packets; B1 starts the delivery timing run | Prove DSP correctness and timing, demo without sensors |
| `LIVE_RAW` | AD7606 only: raw counts, measured rate and error counters on the ST-LINK COM port | Bring up a new ADC, check wiring and channel order |
| `LIVE` | AD7606 → DSP → packets → ESP32, one status line per second | The real system |

Live-mode settings are in [CM7/App/config/live_config.h](CM7/App/config/live_config.h):
speed (`COE_LIVE_NOMINAL_RPM`, any value 60–3600 RPM; now 250 for the lab machine), bearing profile, and **`COE_LIVE_VOLTS_PER_G`** for the
four channels. While these are 0 (unknown), LIVE computes everything from volts
but sends **no packets**, because every packet declares g.

### 2.2 Tools

Everything needed is installed by the VS Code **STM32** extension (GCC, CMake,
Ninja, STM32CubeProgrammer). The scripts in `tools/` find those tools themselves.
Run them from a PowerShell window in this `coe_firmware` folder:

| Script | What it does |
| --- | --- |
| `tools\build.ps1 -Mode <mode>` | Builds both cores into `%LOCALAPPDATA%\coe_firmware\cli-build-<id>` (one per project location), outside the project (VS Code's `CM4/build`, `CM7/build` are not touched) |
| `tools\flash.ps1 -Mode <mode>` | Builds, flashes both cores through the ST-LINK and resets the board (`-NoBuild` to skip building) |
| `tools\com.ps1` | Shows the board's messages from the ST-LINK COM port and saves them under `build\logs` (Ctrl+C stops) |
| `tools\replay_results.ps1` | Reads every REPLAY check from the running board and prints PASS/FAIL (no debugger needed) |
| `tools\ring_snapshots.ps1` | While LIVE_RAW runs, saves the last 136 ms of raw samples every 2 s (a software oscilloscope) |
| `tools\ring_sine_fit.py` | Fits frequency and amplitude in those snapshots; gain table for a generator sweep (section 4.7) |

If Windows refuses to run scripts, put `powershell -ExecutionPolicy Bypass -File`
in front, as in the commands below. Close any VS Code debug session before
flashing with the script (both use the ST-LINK).

VS Code instead: build with the CMake extension, then run "COE: flash both cores
and debug M7" (`.vscode/launch.json`), which flashes and debugs the mode chosen in
`CM7/firmware_mode.cmake`.

### 2.3 Test 1: DSP self-test (REPLAY, no ADC needed)

Switch the 24 V supply off, plug in the STM32 USB, then:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\flash.ps1 -Mode REPLAY
```

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\replay_results.ps1
```

Expect 39 lines, every one PASS or info, ending with "All replay checks passed".
The window times should be about 91 ms (1750 RPM) and 48 ms (3600 RPM), below 333 ms.

### 2.4 Test 2: ADC raw acquisition (LIVE_RAW)

With the 24 V supply off:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\flash.ps1 -Mode LIVE_RAW
```

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\com.ps1
```

Then: switch the 24 V on → press **B1** (the start-up checks print, then
"AD7606 started" and one report per second) → watch for a minute → press **B1**
to stop → 24 V off → Ctrl+C. What to look for: section 4.6. Green LED slow blink
= waiting, fast blink = running; red LED = an error counter is not 0.

### 2.5 Test 3: live chain (LIVE)

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\flash.ps1 -Mode LIVE
```

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\com.ps1
```

Same power order and B1 use as Test 2. Each window prints a line, and each
second a status line (rate, windows, packets, gaps, DSP load, UART). Run 10 minutes
or more for the proof run; gaps, dropped blocks, restarts and stalls must stay 0.
Packets go to the ESP32 only after calibration is set (section 2.1).

### 2.6 Test 4: delivery to the host through the ESP32

Wire STM32 ↔ ESP32 (section 3.3). In the `coe_esp32` folder (ESP COM port: COM6):

```powershell
.\start_esp32.ps1
```

```powershell
idf.py -p COM6 flash monitor
```

In a second PowerShell window, also in `coe_esp32`, start the host receiver:

```powershell
python tests/host_receiver.py --bind 0.0.0.0 --port 8000
```

Flash the STM32 with REPLAY (Test 1) or LIVE (Test 3). In LIVE every window reaches the host
(about 100–150 ms per HTTP hop, 11 Nov 2026). In REPLAY the 12 start-up
packets appear in the ESP monitor and at the host; pressing **B1** once runs the
timed delivery batch (section 7). Wi-Fi: the laptop joins the ESP's own Wi-Fi
**COE-NODE** (or the same hotspot as the ESP, set in the ESP's
`main/config/connection_settings.local.h`). The ESP finds the receiver by itself
(UDP beacon on port 8001), so no laptop IP needs to be set.

### 2.7 Quick ADC check (standalone, minimum wires)

A separate one-file program, `tools/adc_check/adc_check.c`, not part of the
firmware: no HAL, SPI or DSP. Once per second it pulses RESET, sends 10 CONVST
pulses and checks that BUSY answers each with a ~4 µs high. Only these wires:

| STM32 | ADC |
| --- | --- |
| PB8, CN7 pin 2 | RST |
| PD15, CN7 pin 18 | CA (and CB, if joined) |
| PB7, CN10 pin 16 | CB, if wired on its own |
| PB9, CN7 pin 4 | BUSY |
| GND, CN7 pin 8 | GND |

Plus 5V/GND from the converter, and D15, RANGE, OS0–OS2 either to GND or to
their STM32 pins (3.1), which this test also holds low. CS, RD and D7 may stay
connected or be removed.

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\adc_check\run.ps1
```

It builds, flashes the CM7 and shows the result each second. Then switch the
24 V on. Green blinking + "ADC CONVERTS OK" = the chip works. Red = BUSY stuck
high. Yellow = no BUSY pulse (ADC unpowered, or CONVST/BUSY wire). Afterwards,
put the real firmware back (for example `tools\flash.ps1 -Mode LIVE_RAW`).

### 2.8 Software-only checks (no board)

UART sender and packet decoder, with mocked HAL and sanitizers, in Linux/WSL:

```powershell
wsl.exe -d Ubuntu-26.04 --exec python3 '/mnt/c/Users/USER2/OneDrive - KFUPM/Senior Design/coe_firmware/CM7/TestData/communication/run_host_checks.py' --firmware '/mnt/c/Users/USER2/OneDrive - KFUPM/Senior Design/coe_firmware'
```

Decode a saved 160-byte packet with the Python host decoder, from `coe_pipeline`:

```powershell
uv run --no-sync python -m steps.step_08.firmware_payload path_to_packet.bin
```

Regenerate coefficients and test fixtures after changing a DSP setting, from
`coe_pipeline` (one script per stage in `CM7/TestData/generators`), for example:

```powershell
uv run --no-sync python "C:/Users/USER2/OneDrive - KFUPM/Senior Design/coe_firmware/CM7/TestData/generators/generate_bandpass_fixtures.py" --python-project "C:/Users/USER2/OneDrive - KFUPM/Senior Design/coe_pipeline"
```

Then rebuild and repeat Test 1.

## 3. Wiring and power

Switch off the bench supply and unplug USB before changing wires.

### 3.1 AD7606 module to STM32 (CN7)

STM32 board: ST-LINK USB at the top; CN7 is the upper black connector on the right.
ADC module: green terminal block on the left, header P2 on the right. P2 rows are
counted **upward from the bottom**; "left" is nearer the chip.

| ADC pin | P2 position | STM32 CN7 pin / label | GPIO | Function |
| --- | --- | --- | --- | --- |
| RST | row 5 right | pin 2 / D15 | PB8 | Reset pulse |
| BUSY | row 6 left | pin 4 / D14 | PB9 (EXTI9) | Conversion running; falls when results are ready |
| GND | row 1 left or row 7 right | pin 8 / GND | - | Common 0 V |
| RD = SCLK | row 5 left | pin 10 / D13 | PA5 (SPI1_SCK, AF5) | Serial clock |
| D7 = DOUTA | row 11 left | pin 12 / D12 | PA6 (SPI1_MISO, AF5) | Serial data out |
| CS | row 6 right | pin 16 / D10 | PD14 | Chip select |
| CA | row 4 right | pin 18 / D9 (**not** pin 17) | PD15 (TIM4_CH4, AF2) | Conversion start A |
| CB | row 4 left | **CN10 pin 16 / D0** (connector below CN7, right column), or joined with CA on CN7 pin 18 | PB7 (TIM4_CH2, AF2) | Conversion start B; same timer as CA, so both edges coincide |
| OS0 | row 2 right | pin 1 (left column) | PC6, held low | No oversampling |
| OS1 | row 2 left | pin 3 (left column) | PB15, held low | No oversampling |
| OS2 | row 3 right | pin 7 (left column) | PB12, held low | No oversampling |
| **RANGE** | row 3 left | pin 11 (left column) | PC7, held low | ±5 V input range |
| **D15** (BYTE SEL) | row 15 left (top) | pin 20 | PG9, held low | Required low for serial mode |
| 5V | row 1 right | - | - | From the 24→5 V converter, **wired directly** (not via the breadboard) |

Full P2 header (bottom row 1 → top row 15), left / right: 1 GND/5V, 2 OS1/OS0,
3 RANGE/OS2, 4 CB/CA, 5 RD/RST, 6 BUSY/CS, 7 FD/GND, 8–14 D1…D13 / D0…D12,
15 D15/D14. Leave FD, D8 (an output) and the other data pins unconnected.

OS0–OS2, RANGE and D15 must never float (random oversampling, range or read
mode). The firmware holds them low from right after reset; wiring them to GND
on the ADC board instead is equivalent. Most ADC wires use the even (right)
column of CN7; only these configuration pins use the odd (left) column.

Module facts (photos and tests): STBY and PAR/SER are set on the module by its
own 10 kΩ resistors (serial mode confirmed). VDRIVE is about 4.3 V, so its
outputs reach PA6/PB9 at about 4.3 V; **both are 5 V-tolerant (DS12919 Table 8:
PA6 FT_a, PB9 FT_fh)** as long as the STM32 is powered and they are not in
analog mode. The firmware drives RST, SCLK, CS and CONVST **low** right after
reset in every mode, and never drives them high until B1 is pressed.

### 3.2 Analog channels (confirmed wiring)

Since 2 Nov 2026: three TE 805-0050 sensors (the 4th has no cable). Conditioner
output 2 feeds both filter 2 and filter 3, so B2X is a copy of the B1Y sensor
taken before the filters (AIN4 and AIN6 match within about 0.2 mV). Confirmed by
tapping one sensor at a time (4.8). Filter paths stay separate: never tie filter
or ADC outputs together, and never copy one sampled channel in software to
"make" four.

| Sensor (conditioner input) | Conditioner output → filter / TL074 pin | ADC input | Serial slot (V1..V8 = 0..7) | Pipeline channel |
| --- | --- | --- | ---: | --- |
| IEPE 0 | out 1 → OUT1 / pin 1 | AIN2 | 1 | B1X, bearing 1 X |
| IEPE 1 | out 2 → OUT2 / pin 7 | AIN4 | 3 | B1Y, bearing 1 Y |
| IEPE 1 (copy) | out 2 → OUT3 / pin 8 | AIN6 | 5 | B2X, bearing 2 X |
| IEPE 3 | out 4 → OUT4 / pin 14 | AIN8 | 7 | B2Y, bearing 2 Y |

IEPE 2 is empty and conditioner output 3 is not connected. The 805 sensors sit on
a breadboard with jumpers: pin 1 (+OUT) to the BNC center, pin 2 (GND, the case)
to the shield; pin 3 is unused. Powered sensors read about 10.9 V (datasheet
bias 8–12 V).

Path: sensor → conditioner front IEPE input → rear output → filter input (before
its first resistor) → filter output → ADC input. AIN1/3/5/7 are unused.

### 3.3 STM32 to ESP32-C6

| STM32 | Physical | ESP32-C6 | Function |
| --- | --- | --- | --- |
| PD5, USART2 TX | CN9 pin 6 (across from A2) | GPIO18 (RX) | Feature packets |
| PD6, USART2 RX | CN9 pin 4 (across from A1) | GPIO19 (TX) | Delivery acknowledgment |
| GND | CN9 pin 12 (across from A5) | GND | Common 0 V |

Use ESP pins 18/19, not its TX/RX pins. Each board has its own USB cable; do
not connect their 3.3 V or 5 V together.

### 3.4 Power

The SRD-1104 conditioner (+24V IN / GND), the enclosed 24→5 V converter (input
red/black) and the Hi-Link A2415S-1WR3 (pin 1 +Vin, pin 2 −Vin) connect in
parallel to the bench 24 V. Converter output yellow = +5 V → ADC 5V, output
black → common 0 V. Hi-Link (printed face, legs down): pin 4 −15 V → TL074
pin 11, pin 5 COM → breadboard 0 V, pin 6 +15 V → TL074 pin 4. Connect all
grounds (conditioner, filter, ADC, STM32, ESP32, converter returns, Hi-Link
COM) to one 0 V network; **−15 V is not ground**. Leave the conditioner's USB
power disconnected when using 24 V.

STM32 power, two options (JP2 is the jumper cap near the USB connector):

| Option | Wiring | Still noise (11 Nov 2026) |
| --- | --- | --- |
| Laptop USB | JP2 on **STLINK**, nothing on the 5V pin | about 4.3 mg RMS |
| Converter (no laptop) | JP2 on **EXT**, converter 5 V → CN8 pin 9 (5V), GND → CN8 pin 11 | about 12.5 mg RMS |

Never use both at once (JP2 on STLINK with a wire on the 5V pin). VIN needs 7–11 V,
so 5 V cannot go there. Power off: 24 V off **and** STM32 USB unplugged, otherwise the
STM32 signal pins keep the ADC LED dimly lit.

Recorded measurements (DC volts, black probe on the device's 0 V):

| Point | Reading |
| --- | --- |
| Conditioner +24V IN | 24 V |
| Converter output | 5.1 V |
| ADC header 5V | 5.1 V when wired directly (4.6 V through the breadboard: too low, AD7606 needs 4.75–5.25 V) |
| Hi-Link pin 6 / pin 4 vs COM | +15.1 V / −15.1 V; TL074 pins 4 / 11 about +15 / −15.1 V |
| Largest span (+24 to −15.1 V) | about 39.1 V, below the agreed 50 VDC limit (calculated) |
| IEPE 0 excitation | 4.275 mA (cross-check 13.89 V / 3.24 kΩ = 4.287 mA); IEPE 1–3 not yet recorded |

IEPE current check: supply off, meter on DC mA in series: BNC red clip → meter →
3.3 kΩ → BNC black clip, on the conditioner's front input, sensor removed. Do not
use 10 kΩ (4 mA would need 40 V). Still to do: IEPE 1–3, and one test with
everything powered together (voltages under load, total current).

## 4. AD7606 acquisition

### 4.1 How it works

```text
TIM4 CH4 (PD15) --CONVST--> AD7606 converts all 8 inputs (about 4 us)
BUSY falls (PB9) -> EXTI9 interrupt (priority 0): CS low, arm DMA, start SPI1
SPI1 RX -> DMA2 Stream0 -> next ring frame [V1..V8]
DMA complete (priority 1): CS high, SPI off, publish the frame
main loop: whole 256-frame blocks from the ring; never DSP in an interrupt
```

| Item | Setting |
| --- | --- |
| Sample rate | TIM4 200 MHz / 6667 = **29,998.5 S/s per channel**; checked at start (minimum 25,000) |
| SPI1 | Master, receive-only, 16-bit, mode 2, 200 MHz / 32 = **6.25 MHz**; 128 clocks = 20.5 us per read (12.5 MHz possible later) |
| Timing budget | 33.3 us per sample: 4.15 us conversion + 20.5 us read, about 8 us spare |
| Ring | 16 blocks × 256 frames × 8 slots, int16 = 64 KiB in AXI SRAM, about 136 ms of cover |
| Code | `CM7/App/acquisition/` (direct register access; HAL SPI/TIM stay disabled) |

### 4.2 Error counters

Shown on the COM port each second and in Watch (`ad7606_status`, `ad7606_timing`,
`ad7606_stalls`): missed conversions (BUSY gap above 1.5 periods), read overlaps,
ring overruns, DMA/SPI errors, stalls (no frame for 50 ms: acquisition stops and the
pins return low), highest ring fill, CONVST-to-interrupt latency, longest interrupt.
The printed rate counts BUSY edges between cycle-counter timestamps.

### 4.3 Start-up checks (every B1 start)

A 20-ms BUSY timeline (100-ns RESET, CONVST, 50-us RESET, CONVST), a short-circuit
test (each control line driven high alone), pull tests (driven vs floating, CS
working, serial vs parallel mode) and one timed test conversion. Acquisition
starts only if BUSY rises (< 100 ns) and falls (about 4 us) after the test pulse;
otherwise it prints why and keeps the pins low.

### 4.4 LIVE mode

Slots 1/3/5/7 → × (10 V / 65536) / volts-per-g → `pipeline_process_block()` in
the main loop → window callback → `feature_uart_send()`. Any missed conversion,
overlap, overrun, DMA or SPI error drops the affected blocks and restarts the
pipeline, so no window spans missing samples (window IDs restart at 1, which the
ESP accepts). Each window prints RMS and kurtosis; each second prints rate,
windows, packets sent/withheld/failed, gaps, dropped blocks, restarts, ring
fill, longest DSP block, DSP load and UART results (Watch: `live_*`).

### 4.5 Bring-up problems found and fixed (25–26 Oct 2026)

| Symptom | Cause | Fix |
| --- | --- | --- |
| BUSY stuck high, ignores RESET and CONVST | **CB wired to CN7 pin 17 (PA4) instead of pin 18**: CB floated, so the chip waited forever for the second conversion start | CA and CB joined (same breadboard row) on CN7 pin 18 |
| AD7606 supply 4.6 V (needs 4.75–5.25 V) | 0.5 V lost through the breadboard | Converter wired straight to the module's 5V/GND |
| Undefined read mode / input range | D15 (BYTE SEL) and RANGE floating | Both to GND |
| BUSY high after every RESET until the next CONVST (normal for this chip) | - | Start-up sends one CONVST to clear it |
| Some samples read about 2× (one-bit shift, worse for later slots) | Ringing on the long SCLK jumper counted as an extra clock | SCLK, CS and CONVST pins set to the slowest edge rate; none since |

Four-channel run (26 Oct 2026), filter OUT1 on AIN2/4/6/8: rate and counters
perfect, the four slots agree sample by sample within about 1 mV. The signal
itself is not vibration-like: mean about 1.0 V, ceiling about 1.50 V, and in the
last 136 ms of raw samples (read from the ring over the ST-LINK) clean steps
between about 0.8 V and 0 V lasting 5–20 ms; taps were not visible in the
per-second statistics. Next: check the analog path (loose contact, conditioner
or filter state), for example by reading the ring during still and tapping periods.

The standalone check (2.7) gives the quickest yes/no: on this module, 10/10 pulses
answered with BUSY high for 4125 ns (datasheet conversion time). With all inputs
unconnected every slot reads about 1.75 V with about 10 mV of noise (the module's
open-input level). If doubled values return, shorten the SCLK wire or add about
100 Ω in series at the STM32 end.

### 4.6 Bring-up checklist (any module)

Steps 1–3 passed with the current module on 26 Oct 2026.

1. Wire as in 3.1 (OS0–2, RANGE, D15 to their STM32 pins or GND, 5 V wired directly, CA on CN7 pin 18,
   CB on CN10 pin 16 or joined with CA, jumpers pushed fully on). **Leave the filter outputs disconnected**
   for steps 2–3. Optional first: the standalone check (2.7).
2. Flash `LIVE_RAW` and open the COM port (commands: section 2.4); USB → 24 V → B1.
   Expect "AD7606 started" and BUSY pulses.
3. Run a minute: rate about 29,998.5 Hz, missed/overlap/overrun/dma/spi all 0,
   no slot jumping to about twice its value.
4. 24 V off, connect the filter outputs, 24 V on, B1 again: it must still start.
   Put a signal on one input at a time: AIN2/4/6/8 must move slots V2/V4/V6/V8 only.
   Note what the unused inputs read (near 0 and quiet = grounded).
5. Flash `LIVE`, run 10 minutes or more with the ESP32: gaps, dropped blocks,
   restarts and stalls stay 0. Record ring fill, longest block and DSP load.
6. With calibration set, check packets arrive at the host; record results here.

### 4.7 Measured response: generator → filter → ADC (26 Oct 2026)

FNIRSI 2C53T sine, 1.0 V peak-to-peak (it outputs 0–1 V), into filter IN1;
filter OUT1 to all four ADC inputs; LIVE_RAW running. The generator alone into
the ADC gave 0.993 V at 100 Hz. To repeat: while LIVE_RAW runs, save the raw
sample ring every 2 s, then fit a sine to each snapshot (from `coe_pipeline`):

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\ring_snapshots.ps1 -Minutes 10
```

```powershell
uv run --no-sync python ../coe_firmware/tools/ring_sine_fit.py ../coe_firmware/build/logs/ring --steps 100 1000 5000 8000 10000 12000 14000
```

| Input | Output (V pp) | Gain | dB |
| ---: | ---: | ---: | ---: |
| 100 Hz | 0.993 | 1.000 | 0.0 |
| 1 kHz | 0.991 | 0.998 | −0.02 |
| 5 kHz | 0.944 | 0.950 | −0.44 |
| 8 kHz | 0.827 | 0.832 | −1.59 |
| 10 kHz | 0.702 | 0.706 | **−3.02** |
| 12 kHz | 0.561 | 0.565 | −4.95 |
| 14 kHz | 0.434 | 0.437 | −7.19 |

- The −3 dB point is at **10 kHz**, as designed; the response is smooth (fit
  error a few mV), with no distortion or oscillation, and all four slots agree.
- The response includes the AD7606's own input filter and assumes the
  generator is flat over this range (only 100 Hz was measured directly).
- Why there is loss below the filter's 11.5-kHz design cutoff: a 2nd-order
  low-pass is already −3 dB at its cutoff and rolls off gradually below it
  (ideal 11.5-kHz Butterworth: −0.9 dB at 8 kHz, −2.0 dB at 10 kHz), and the
  AD7606's built-in 2nd-order anti-alias filter adds more. Fitting "2nd-order
  filter × first-order 15-kHz roll-off" to the seven points gives the board
  filter **f0 = 11.6 kHz, Q = 0.75** (Butterworth 0.71), within 0.01 dB of every
  measurement: the built filter matches its design.
- In the 2–8 kHz DSP band the gain falls from about 0.99 to 0.83. A fixed gain
  shape like this can be corrected later with calibration if absolute g
  levels near 8 kHz matter.
- At 14 kHz the signal is only −7 dB: a 2nd-order filter at 10 kHz gives modest
  protection against aliasing at 30 kS/s. Still to measure: 20–28 kHz inputs,
  which would fold into the 2–8 kHz band.
- During the whole 133-s run: 29,998.50 S/s, 0 missed/lost samples.

### 4.8 Sensor tap test (2 Nov 2026)

LIVE_RAW, per-second report plus ring snapshots (2.4, 4.7). Before the test the
old bench supply smoked inside at switch-on; the circuit's 24 V input showed no
short (about 1.9 MΩ) and everything worked on a replacement supply.

First run: AIN2 and AIN8 drifted by volts as exact mirror images (correlation
−0.97 to −0.99, sum constant at 4.62 V), and levels sat at 0.6–3 V. Cause: the
Hi-Link A2415S-1WR3 is isolated, and its COM (the filter's 0 V rails) was not
connected to the circuit ground (no beep; +15 V read 13 V from circuit ground).
One jumper from the breadboard 0 V rail to the 24 V return fixed it (3.4: all
grounds on one 0 V network). If channels drift or sit at odd DC levels, check
this link first.

After the fix (quiet, 30 s): 0.3 mV RMS noise on every channel, means 3–7 mV.
Taps, one sensor at a time: IEPE 0 → AIN2 only, IEPE 1 → AIN4 and AIN6, IEPE 3 →
AIN8 only. Table knocks move all channels together (the sensors share one
breadboard). A direct pen tap on a sensor clips at ±5 V (±50 g, the 805-0050
range) and the level then recovers smoothly in under a second; that is overload,
not a bad contact. No square jumps were seen.

LIVE with the sensors (calibrated, 0.100 V/g), 107 s, 156 windows: all sent,
gaps 0, dropped blocks 0, UART errors 0, worst latency 16.08 ms (limit 333),
DSP load about 16 %. Still: RMS about 1.2 mg on every channel, kurtosis about 3.
Taps: IEPE 0 → B1X 0.9–1.6 g; IEPE 1 → B1Y and B2X 0.4–1.3 g; IEPE 3 → B2Y
0.8–1.0 g. Moving the breadboard: 20–120 mg on all channels, kurtosis 5–10.
Open: while still, B1Y/B2X showed small spikes (about 3 mV, every ~2 s) while
`ring_snapshots.ps1` was reading the board over SWD every 2 s; repeat a still
run without it. ESP32 reception and host delivery with the sensors not yet run.

## 5. DSP pipeline

Input: four channels in g at 30 kS/s, order B1X, B1Y, B2X, B2Y. One window =
20 nominal revolutions: 20,571 samples at 1750 RPM, 10,000 at 3600 RPM. Any
speed from 60 to 3600 RPM is accepted. Below about 1500 RPM, 20 revolutions do
not fit the envelope FFT, so a window is the most whole revolutions that fit
(250 RPM: 3 revolutions, 21,600 samples, 0.72 s). The fault bands are never
narrower than one FFT bin (0.10 order, or 1.22 Hz at slow speeds). Windows are not rounded to 256-sample blocks; filter
and decimation state carry across blocks and windows.

| Step | Module (`CM7/App/dsp`) | Definition |
| --- | --- | --- |
| Band-pass | `bandpass` | Butterworth order 8 (4 biquads), 2–8 kHz, causal, CMSIS DF2T |
| Time features | `time_features` | Signed RMS (no mean removal), Pearson kurtosis (Gaussian = 3), crest factor; one-pass centred moments in double |
| Envelope | `envelope` | Absolute value → Butterworth order 7 low-pass (0.1 dB to 350 Hz, ≥ 60 dB from 1250 Hz) → keep 1 in 6 (5 kS/s) |
| Spectrum | `envelope_spectrum` | Whole-window envelope: remove mean, periodic Hann of the actual length, zero-pad to 4096, CMSIS real FFT; 1.2207 Hz bins; amplitude and one-sided PSD |
| Frequency features | `frequency_features` | Envelope RMS (with mean), plus PSD band RMS at FTF/BSF/BPFO/BPFI orders ± 0.10 order |

Bearing profiles (orders): **project, ID 1, provisional**: FTF 0.38, BSF 1.98,
BPFO 3.05, BPFI 4.95. CWRU drive end, ID 2: 0.39828, 2.35675, 3.5848, 5.4152.
Confirm the project orders for the real bearing before AI training.

API rules (`CM7/App/pipeline/pipeline.h`): call `pipeline_init()` once per
stream with static envelope buffer (55 KB) and FFT workspace (63 KB);
`pipeline_process_block()` takes any block size, from the main loop only; a
false return invalidates the stream (re-initialize). The window callback borrows
its result: copy what you keep before returning. Constant/zero channels get
zero features and a cleared validity bit, which is not a "healthy" label.

Coefficients and test fixtures are generated from `coe_pipeline` by
`CM7/TestData/generators/*.py` (run with `uv run --no-sync python ...` from
`coe_pipeline`). Change settings, generator and fixtures together.

## 6. Feature vector and packet

32 Float32 values, channel-major (B1X 0–7, B1Y 8–15, B2X 16–23, B2Y 24–31).
Per channel: 0 band-pass RMS (g), 1 kurtosis, 2 crest factor, 3 envelope RMS (g),
4 FTF band RMS, 5 BSF band RMS, 6 BPFO band RMS, 7 BPFI band RMS (g).

Packet, schema 1, 160 bytes, little-endian:

| Bytes | Field |
| --- | --- |
| 0–3 | `COE1` |
| 4–5 / 6–7 | Schema 1 / feature count 32 |
| 8–11 | Window ID (from 1 after each stream start) |
| 12–15 | Nominal RPM |
| 16–19 | Bearing profile ID |
| 20–23 | Channel-valid mask (bit 0 B1X … bit 3 B2Y) |
| 24–27 | Unit ID, 1 = g |
| 28–155 | 32 features |
| 156–159 | CRC-32 (zlib.crc32) of bytes 0–155 |

Host decoder: `coe_pipeline/steps/step_08/firmware_payload.py`. Use the same
features, units and normalization for training and inference.

## 7. STM32 to ESP32 link

USART2, 115200 8N1, TX by DMA1 Stream0 (interrupt priority 6), two owned
160-byte slots in AXI SRAM (`CM7/App/communication/feature_uart.c`). Call
`feature_uart_send()` and `feature_uart_process()` from the main loop only. A
full queue returns false (never overwrites); an error or 100-ms timeout stops
the transmitter until reset. One packet takes 13.9 ms on the wire. The ESP
validates and forwards packets over HTTP; see `../coe_esp32/README.md`.

Delivery timing (REPLAY mode, `delivery_timing.c`): after both boards and the
host receiver run, press B1 once: twelve windows are processed, each waiting for
the host acknowledgment via ESP GPIO19 → PD6. Watch `delivery_timing_results`
(processing, UART, ESP queue, network and round-trip times),
`delivery_timing_completed` (expect 12) and `delivery_timing_timeouts` (0).
Commands for the ESP side and the host receiver: section 2.6.

## 8. Verification and Watch variables

Saved board results (REPLAY, 400 MHz; re-run on the HSE clock 25 Oct 2026, all passed):

| Check | Cases | Result |
| --- | ---: | --- |
| Time features | 48 channel-windows | passed |
| Band-pass | 16 runs | passed |
| Envelope | 16 runs + 4 windows | passed |
| FFT (all 2049 bins) | 9 records | passed, max amplitude error 3.7e-6 g |
| Frequency features | 18 | passed |
| Streaming / batch pipeline | 12 / 12 windows | passed |
| Packet and decoder | 22 decoder cases + exact bytes | passed |
| Complete window time, 1750 / 3600 RPM | | **91.03 / 47.87 ms** (limit 333 ms) |
| Longest block call (window close, 4 FFTs) | | 10.68 ms (blocks arrive every 8.53 ms: the ring covers it) |

Presentation Watch list (REPLAY): `SystemCoreClock` (400000000),
`processing_clock_source` (1 = HSE), `pipeline_batch_max_ms_1750`,
`pipeline_batch_max_ms_3600`, `pipeline_batch_deadline_met`,
`pipeline_batch_checks_passed`, `pipeline_batch_features_1750` /
`final_pipeline_expected[1][0][2]` (MCU vs Python), `feature_uart_ready`,
`feature_uart_packets_sent` (12 after start-up), `feature_uart_errors` (0).
Every `*_checks_passed` should be true and every `*_failed_check` 0; a failure
code points to a case in `CM7/TestData/demos`. Tolerances: 2e-5 × (1 + |ref|)
for time features, 4e-5 × (1 + |ref|) for envelope/frequency features. Do not
pause the debugger inside a timed run. Raw results: `CM7/TestData/results/final_board_results.json`.
`tools\replay_results.ps1` reads all of these from the board at once (section 2.3).
The host-only UART/decoder checks (section 2.7) use mocked HAL headers in
`CM7/TestData/communication/host`, which are never part of the firmware build.

## 9. Clock and memory

`configure_processing_clock()` (`CM7/App/platform/clock`) runs before the M4 is
released: direct SMPS, VOS1, PLL1 from HSE bypass 8 MHz (M=2, N=200, P=2 → 400 MHz
M7, Q=4 → 200 MHz SPI1 kernel), AHB 200 MHz, all APB 100 MHz (timers 200 MHz),
Flash 4 wait states, instruction cache on, data cache off. If HSE fails or its
measured core clock is more than 3% off (checked against HSI with LPTIM1), it falls
back to HSI 64 MHz (M=4, N=50). `HSE_VALUE` = 8 MHz is set in both cores'
`CMakeLists.txt`. 480 MHz would need VOS0 and an LDO board change: not used.

| Mode | Flash | DTCM (128 KB) | AXI SRAM (512 KB) |
| --- | ---: | ---: | ---: |
| REPLAY | 861,652 B | 107,824 B | 494,112 B (replay buffers) |
| LIVE_RAW | 43,588 B | 9,408 B | 65,536 B (ring) |
| LIVE | 150,144 B | 78,392 B | 120,736 B (ring + envelopes) |

DMA cannot reach DTCM, so every DMA buffer is in AXI SRAM. The M4 uses D2 SRAM.

## 10. Project layout and build notes

```text
CM7/App/
  config/         pipeline_config.h (rates, sizes), firmware_mode.h, live_config.h
  acquisition/    AD7606 driver, safe pins, console, LIVE_RAW monitor, LIVE stream
  pipeline/       pipeline coordinator (windows, callback)
  dsp/            bandpass, time_features, envelope, envelope_spectrum, frequency_features
  payload/        32-feature vector and 160-byte packet
  communication/  USART2 DMA sender and delivery timing
  platform/       clock setup, cycle-counter timing
CM7/TestData/     REPLAY-only demos, fixtures, generators, saved results, host tests
tools/            build, flash, COM-port and REPLAY-results scripts (section 2.2)
CM7/Core/         CubeMX code (main.c: edit only inside USER CODE sections)
CM7/cmake/        CMSIS-DSP source list (kernels in Drivers/CMSIS/DSP, unmodified)
CM4/              idle second core
Drivers/          ST HAL, BSP and CMSIS (vendor code)
```

- Each `.h` explains its module's contract; read `pipeline_config.h`, then
  `pipeline.h`, then the DSP headers in order.
- `mx-generated.cmake` and the root `CMakeLists.txt` are CubeMX files: do not edit.
  The root ExternalProject does not pass `-D` options to CM7, which is why the mode
  is a file. After a CubeMX regeneration, check `CM7/CMakeLists.txt`, the linker
  sections (`.pipeline_test_buffers`, `.feature_uart_dma`, `.pipeline_live_buffers`,
  `.acquisition_dma`) and keep USART2, DMA1/2, SPI1, TIM4 and EXTI9 unassigned in CubeMX.
- Production DSP and the AD7606 driver are compiled with `-O2` even in Debug; CMSIS
  kernels with `-Ofast`. No fast-math in application code.
- The CM7 build also compiles `../coe_esp32/main/transport/latency_packet.c`
  (all modes) and `feature_packet.c` (REPLAY decoder checks); keep the two
  folders side by side.
- Git: `main` and tag `working-hotspot-2026-10-05` hold the hotspot baseline;
  `codex/layer-latency` holds the timing and acquisition work. Switch `coe_firmware`
  and `coe_esp32` together, then rebuild and flash both boards. ESP Wi-Fi
  credentials live in the ignored `main/config/connection_settings.local.h`.

## 11. Open items

- Sensors: mount them rigidly (adhesive or stud, as in the 805 datasheet) on the
  rig instead of the breadboard; get a cable for the 4th sensor and remove the
  B2X copy.
- Calibration: set to the 805-0050 nominal 0.100 V/g (2 Nov 2026); use the
  per-serial calibration sheet if one turns up.
- Confirm the real bearing's orders (speed is now set per machine in `live_config.h`).
- On the machine: record about 1 minute with the machine off (noise floor baseline), then the runs.
- Aliasing suppression: measure 20–28 kHz generator inputs (4.7).
- IEPE 1–3 current and the all-powered-together power test.
- Warm-up after start, and revolution-accurate windows (needs a shaft reference).
