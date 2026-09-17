# COE Processing Pipeline Component Details

## Purpose

This document explains what each STM32 software component does, what data it receives, and what it produces.

The design is intentionally limited to what the two-month prototype needs. It does not add a complicated communication protocol, a large status system, or calculations that are not required by the AI model or FE digital twin.

The main goals are:

- Acquire four acceleration signals and two shaft-displacement signals together.
- Use the Keyphasor to define one 20-revolution measurement window.
- Produce 32 accelerometer features for the AI model.
- Produce two displacement amplitude/phase pairs for the FE digital twin.
- Send one fixed 152-byte payload for every valid window.

---

## 1. Fixed Design Values

| Item | Selected value |
| :--- | :--- |
| Active analog inputs | Four accelerometers and two radial proximity probes |
| ADC | AD7606 with eight simultaneous channels |
| AD7606 oversampling | Disabled, `OS[2:0] = 000` |
| Selected output rate | 30,000 sample sets per second |
| Serial read clock | 10 MHz |
| Operating modes | 1750 RPM and 3600 RPM, steady operation |
| DMA block size | 256 simultaneous sample sets |
| One DMA block | 4,096 bytes |
| Two alternating DMA blocks | 8,192 bytes total |
| Time to fill one block at 30 kSamples/s | 8.533 ms |
| AI/FE measurement window | 20 shaft revolutions |
| Envelope rate | 5,000 samples/s after decimation |
| Envelope-spectrum transform | Hann-windowed record, zero-padded to a 4,096-point FFT |
| Features | Eight per accelerometer, 32 total |
| Bearing orders | FTF 0.38, BSF 1.98, BPFO 3.05, BPFI 4.95 |
| Displacement results | Amplitude and phase for two probes, four values total |
| Application payload | 152 bytes |

The STM32 generates `CONVST` at 30 kHz with AD7606 oversampling disabled. Each trigger produces one simultaneous result per channel, so the firmware and DMA receive 30,000 sample sets per second. The custom four-channel fourth-order MFB low-pass filter provides the primary approximately 10 kHz analog anti-aliasing for accelerometer channels V1–V4, while the AD7606's built-in second-order analog filter remains supplemental. The reduction to the 5 kSamples/s envelope rate still uses a decimation factor of 6. Component 3 initially uses a 2–8 kHz band-pass, which is a project baseline for the custom 6200 bearing assembly rather than a universal resonance band for every 6200 bearing. Digital filter coefficients must be calculated for the unchanged 30 kSamples/s rate.

The rotor operates in two steady-speed modes: 1750 RPM and 3600 RPM. Windows acquired during startup, shutdown, or transitions between modes are not used for the baseline AI/FE output.

---

## 2. Hardware Before the Software Pipeline

The analog signals reach the STM32 through these paths:

```text
Four accelerometers
  -> SRD-1104 IEPE conditioner
  -> four-channel fourth-order 10 kHz MFB LPF
  -> AD7606 channels V1-V4
     (supplemental on-chip analog filter; oversampling disabled)

Two RK4 radial proximity probes
  -> RK4 Proximitor units
  -> voltage-compatible protection stage
  -> AD7606 channels V5-V6
     (on-chip analog filter; oversampling disabled)

RK4 Keyphasor
  -> signal conditioning
  -> STM32 timer-capture input
```

The SRD-1104 is used for the IEPE accelerometers. Its four conditioned outputs pass through the custom fourth-order MFB low-pass filter before reaching AD7606 channels V1–V4. The external filter has an approximately 10 kHz cutoff and is the primary anti-aliasing stage for those channels. The proximity probes use their Proximitor units and are not routed through the IEPE conditioner or the four-channel accelerometer LPF.

AD7606 digital oversampling is disabled with `OS[2:0] = 000`. The ADC's built-in second-order analog filter remains active as supplemental input filtering. Bench verification must measure the complete SRD-1104 → external LPF → AD7606 response, confirm that the required 2–8 kHz band is preserved, and verify attenuation above the intended passband. See the [Analog Devices AD7606 datasheet](https://www.analog.com/media/en/technical-documentation/data-sheets/ad7606_7606-6_7606-4.pdf) and [LPF Design Details](LPF-design-details-circuit.md).

The planned AD7606 mapping is:

| ADC channel | Signal |
| :---: | :--- |
| V1 | Plane 1 X acceleration |
| V2 | Plane 1 Y acceleration |
| V3 | Plane 2 X acceleration |
| V4 | Plane 2 Y acceleration |
| V5 | Plane 1 radial displacement |
| V6 | Plane 2 radial displacement |
| V7 | Unused/reserved |
| V8 | Unused/reserved |

The wiring names must be confirmed during assembly, but the software order will remain fixed after that confirmation.

---

## 3. Where Blocks Become a Complete Window

The DMA blocks are only small containers used to move data safely through memory. They are not separate AI windows.

Components 1 through 5 process the incoming data block by block. Component 5 keeps the reduced envelope samples and the running statistics that belong to the current 20-revolution window. Component 8 separately accumulates the proximity information.

When the twentieth Keyphasor revolution ends, the logical window closes. Components 6 through 9 then finalize one result for the complete window.

The raw DMA blocks are never joined into one large raw buffer.

| Component | Data handled while acquiring | Result after 20 revolutions |
| :--- | :--- | :--- |
| 1. Acquisition and window formation | 256-sample DMA blocks | Window timing, RPM, and Keyphasor boundaries |
| 2. Calibration | One DMA block at a time | Calibrated values have contributed to the window |
| 3. Bearing-resonance band-pass | One accelerometer block at a time | Running vibration statistics |
| 4. Rectification | One filtered block at a time | Data passed immediately to Component 5 |
| 5. Envelope low-pass and decimation | One rectified block at a time | Four reduced envelope records covering the window |
| 6. Envelope spectrum and order mapping | Complete time-envelope records | Four frequency spectra with order coordinates |
| 7. Feature extraction | Some statistics accumulated earlier | One 32-feature vector for the complete window |
| 8. Displacement 1x extraction | Updated as blocks/revolutions arrive | Two amplitude/phase pairs |
| 9. Payload construction | Starts after both branches finish | One 152-byte payload |

---

## 4. Simple Firmware Mental Model

The nine components are **logical processing stages**. They explain what must happen to the data, but they do not require nine separate programs, tasks, classes, or source files.

A simple firmware implementation can group them into approximately four main functions:

```text
process_dma_block()
  -> Component 2: calibrate the six active channels
  -> Component 3: band-pass the accelerometers
  -> Component 4: rectify the filtered acceleration
  -> Component 5: smooth and decimate the envelope

process_proximity_revolution()
  -> Component 8: update the two 1x displacement calculations

finalize_window()
  -> Component 6: envelope-spectrum FFT and order mapping
  -> Component 7: create the 32-feature vector

build_and_send_payload()
  -> Component 9: create and send the 152-byte payload
```

Component 1 is mainly the STM32 timer, ADC, DMA, and Keyphasor interrupt handling that supplies data to these functions.

The runtime sequence is:

1. During acquisition, Component 1 repeatedly supplies DMA blocks to `process_dma_block()`.
2. Whenever a Keyphasor edge completes a revolution, `process_proximity_revolution()` updates the displacement calculation.
3. When the twentieth revolution ends at `t20`, `finalize_window()` produces the final 32 accelerometer features.
4. After the accelerometer and proximity results are ready, `build_and_send_payload()` creates the message for the ESP32-C6.

These function names are illustrative; the firmware team can rename them. The important point is that the nine design stages can be implemented with a small and understandable software structure.

The selected design uses all nine stages. The envelope-decimation factor in Component 5 is 6 for the 30-to-5 kSamples/s reduction; the envelope low-pass operation must precede this reduction.

---

# Component 1: Synchronized Acquisition and Window Formation

## Purpose

Component 1 collects the six active analog signals through the eight-channel ADC and determines which samples belong to the same 20-revolution window.

## Operation

1. Set the AD7606 oversampling pins to `OS[2:0] = 000`, disabling digital oversampling.
2. An STM32 hardware timer triggers `CONVST` 30,000 times per second.
3. For each trigger, the AD7606 performs one simultaneous conversion of V1 through V8 and produces one eight-word result.
4. When `BUSY` falls, the STM32 starts a 10 MHz SPI/DMA read of that completed result. Reading may continue while the next conversion is running.
5. DMA places each eight-word ADC sample set into the active 256-sample block. V7 and V8 are retained as unused/reserved words so the transfer format remains simple.
6. While DMA fills one block, the processor handles the other block.
7. A timer-capture input records every Keyphasor edge.

Each ADC sample set contains:

> 8 channels × 2 bytes = 16 bytes

Each DMA block therefore contains:

> 256 sample sets × 16 bytes = 4,096 bytes

At 30,000 sample sets per second, one block covers:

> 256 ÷ 30,000 ≈ 0.008533 seconds = 8.533 ms

## ADC Conversion and Read Timing

One `CONVST` period at 30 kHz is:

> 1 ÷ 30,000 = 33.333 microseconds

With oversampling disabled, the AD7606 maximum conversion time is 4.2 microseconds. A single-lane serial read contains 128 bits, so at 10 MHz it takes:

> 128 ÷ 10,000,000 = 12.8 microseconds

Conversion and the serial read require at most approximately 17.0 microseconds together, leaving approximately 16.3 microseconds before the next `CONVST`. The result must also be read before the following `BUSY` falling edge replaces the output register. Successive results are 33.333 microseconds apart, leaving approximately:

> 33.333 − 12.8 = 20.533 microseconds

for `BUSY`-interrupt latency, starting SPI/DMA, and timing margin. A read must not occur exactly at the `BUSY` falling edge when the output register is updated.

This does not mean the complete pipeline must finish every 8.533 ms. It means the streaming work for one block must finish before that memory block is needed again. Because there are two alternating blocks, DMA continues acquiring while the processor works.

## Forming the Window

One revolution is the interval between two Keyphasor edges. A complete 20-revolution window therefore has:

- One starting Keyphasor edge.
- Twenty following revolution intervals.
- One ending edge after revolution 20.

The first boundary is called `t0` and the final boundary is called `t20`.

The final boundary can occur in the middle of a DMA block. Samples before `t20` belong to the current window. Samples at or after `t20` belong to the next window.

## Window Duration

Window duration depends on speed. RPM divided by 60 gives revolutions per second. Dividing the 20 revolutions in a window by that value gives its duration:

> Window duration = 20 ÷ (RPM ÷ 60) = 1,200 ÷ RPM seconds

The value 1,200 comes from 20 revolutions × 60 seconds/minute.

| Speed | Time for 20 revolutions | Approximate samples per channel at 30 kSamples/s |
| :---: | :---: | :---: |
| 1750 RPM | 685.7 ms | 20,571 |
| 3600 RPM | 333.3 ms | 10,000 |

Average RPM is measured for each window rather than copied from the nominal speed setting. It is calculated as:

> Average RPM = 1,200 ÷ measured window duration in seconds

The value 1,200 comes from 20 revolutions × 60 seconds/minute.

## Data Retained

Component 1 retains only:

- The current DMA blocks.
- The Keyphasor boundary times.
- The window start and end times.
- The number of samples received.
- The average RPM.

If DMA data is lost, a Keyphasor boundary is missing, or speed is outside the accepted steady-operation limits around the selected 1750 RPM or 3600 RPM mode, the window is discarded rather than sent as valid data. The allowed deviation from the nominal speed and variation within a window are set during commissioning; exact equality to the nominal RPM is not required.

## Output

- Successive 256-sample blocks containing six active synchronized signals and two reserved ADC words.
- Keyphasor timing associated with the current window.
- Average RPM and the `t20` window-end timestamp.

---

# Component 2: Channel Calibration and Signal Preparation

## Purpose

The AD7606 produces integer ADC codes. Component 2 converts them into meaningful acceleration and displacement values.

## Operation

For every DMA block:

1. Apply the fixed V1–V8 channel mapping.
2. Convert each ADC code to voltage using the selected AD7606 input range.
3. Apply the stored offset and scale for each channel.
4. Convert V1–V4 into acceleration, normally in m/s².
5. Convert V5–V6 into shaft displacement, normally in micrometres.

The accelerometer conversion uses the sensitivity supplied with the selected accelerometer and IEPE conditioner. The proximity conversion uses the sensitivity of the installed probe/Proximitor system.

No calibration values are transmitted in every payload. They are stored in the STM32 configuration and confirmed during setup.

## Branch Point

After calibration, the data divides into two paths:

- Four acceleration channels go to Component 3.
- Two displacement channels go to Component 8.

## Output

- Four calibrated acceleration streams.
- Two calibrated displacement streams.

---

# Component 3: Bearing-Resonance Band-Pass Filter

## Purpose

A bearing defect can create repeated impacts. Those impacts excite a higher-frequency mechanical resonance in the bearing, housing, and sensor mounting.

Component 3 keeps this useful frequency region and reduces unrelated low- and high-frequency vibration.

## Operation

- Only the four accelerometer channels enter this component.
- The digital band-pass filter operates at the original 30 kSamples/s rate.
- The same filter settings are used for all windows after commissioning.
- The filter state continues across DMA block boundaries. It is not restarted for every block.

The baseline passband is **2–8 kHz**, based on the mechanical team's current estimate for the custom 6200 bearing assembly and the intended sensor range. This is the closest usable design value before measurements exist. It is not a universal 6200-bearing resonance band: the custom block, mounting and sensor affect the response. If measured vibration or a kurtogram later identifies a more impulsive band, update the filter coefficients without changing the rest of the pipeline.

This decision follows Randall and Antoni's bearing-diagnostics tutorial: bearing impacts excite high-frequency resonances of the structure between the bearing and transducer, while the appropriate demodulation band must be selected for the installed case. The tutorial describes spectral kurtosis and the kurtogram as ways to locate the band where impulsiveness is strongest. See [Randall and Antoni, *Mechanical Systems and Signal Processing*, 2011](https://doi.org/10.1016/j.ymssp.2010.07.017).

## Statistics Accumulated Here

While each filtered block passes through, the firmware updates running values needed later for:

- Band-pass RMS.
- Band-pass kurtosis.
- Band-pass crest factor.

This avoids storing the complete high-rate filtered waveform.

## Output

- Four signed band-pass-filtered acceleration blocks.
- Running statistical totals belonging to the current 20-revolution window.

---

# Component 4: Envelope Extraction by Full-Wave Rectification

## Purpose

The band-pass waveform still changes rapidly between positive and negative values. The defect information is mainly contained in how the strength of that waveform rises and falls.

Component 4 makes both halves positive so that Component 5 can recover the smooth amplitude pattern.

## Operation

For every accelerometer sample:

> Rectified value = absolute value of the band-pass sample

Examples:

- `+3.0` becomes `3.0`.
- `−3.0` also becomes `3.0`.

The result is not yet the final smooth envelope. It still contains high-frequency ripple.

## Output

- Four rectified acceleration blocks passed directly to Component 5.

---

# Component 5: Envelope Low-Pass Filtering and Decimation

## Purpose

Component 5 smooths the rectified signal and reduces its sample rate. The result keeps the slower bearing-impact pattern needed for order analysis.

## Operation

1. Apply an envelope low-pass filter to remove the fast ripple remaining after rectification.
2. Keep every sixth filtered sample for the reduction from 30 kSamples/s to the 5 kSamples/s envelope rate: 30,000 ÷ 6 = 5,000 samples/s.
3. Append the retained envelope samples to the current 20-revolution envelope record.
4. Update the running envelope RMS calculation.

The low-pass filter must run before decimation so frequencies above the new 2.5 kHz Nyquist limit do not alias into the reduced envelope.

The filter state and decimation count continue across block boundaries. They are not restarted for each DMA block.

## Data Retained for the Window

At the 5 kSamples/s envelope rate:

| Speed | Envelope samples per channel over 20 revolutions |
| :---: | :---: |
| 1750 RPM | Approximately 3,429 |
| 3600 RPM | Approximately 1,667 |

At 1750 RPM, four Float32 envelope records require approximately:

> 3,429 samples × 4 channels × 4 bytes = 54,864 bytes

Buffer capacity must also allow for the minimum accepted speed and boundary samples; these nominal counts are approximate.

This is much smaller than storing all ADC words for the complete window.

## Output

After `t20`, Component 5 provides:

- Four reduced envelope records covering the complete 20 revolutions.
- One completed envelope-RMS value per accelerometer channel.

This is the end of the main block-by-block accelerometer processing. Component 6 begins the complete-window operations.

---

# Component 6: Envelope Spectrum and Order Mapping

## Purpose

Component 6 converts each completed time-sampled envelope into a frequency spectrum, then uses the measured RPM to express each frequency location as shaft order.

- Order 1 means once per shaft revolution.
- Order 2 means twice per revolution.
- Order 4.95 means 4.95 times per revolution.

During each window the rotor is settled at either 1750 or 3600 RPM. At steady speed, time and shaft angle are related linearly, so angular resampling is not needed to identify orders. A measured spectral frequency is converted using:

> Order = frequency in hertz ÷ (measured RPM ÷ 60)

## Operation

For each of the four time-envelope records from Component 5:

1. Use the actual samples belonging to the 20-revolution window: approximately 3,429 samples at 1750 RPM or 1,667 samples at 3600 RPM.
2. Remove the average value so the large envelope DC component does not dominate the spectrum.
3. Apply a Hann window to the actual record to reduce spectral leakage.
4. Append zeros until the array contains 4,096 values.
5. Calculate a 4,096-point real FFT and consistently scaled single-sided magnitudes.
6. For FFT bin `k`, calculate `frequency[k] = k × 5,000 ÷ 4,096` hertz.
7. Convert each frequency to order using the measured average RPM from Component 1.

Zero-padding supplies a convenient fixed FFT size and denser frequency samples. It does not add measured information or improve the physical resolution set by the record duration.

Because every record covers exactly 20 revolutions, its natural order resolution is:

> Natural order resolution = 1 ÷ 20 = 0.05 order

## Warning: When Angular Resampling Must Be Added

Angular resampling is not part of the baseline design. Component 1 still retains the 21 Keyphasor times and can calculate the duration of every revolution.

If later Keyphasor measurements show meaningful speed change inside a window, and the direct envelope spectra show broadened rotation-related peaks or unstable band features, add computed order tracking between Components 5 and 6. It would resample the envelope at equal shaft-angle intervals before the FFT. Supporting two different settled speeds by itself does not require that operation.

This is an application-specific simplification, not a claim that order tracking is generally unnecessary. Randall and Antoni describe angular resampling as a method for preventing rotation-related spectral components from smearing when shaft speed fluctuates. Because this baseline accepts only settled, constant-speed windows, the direct FFT plus measured-RPM conversion is sufficient unless testing reveals that failure condition. See [Appendix B of Randall and Antoni, 2011](https://doi.org/10.1016/j.ymssp.2010.07.017).

## Output

- Four envelope frequency spectra with a corresponding order value for every bin.

---

# Component 7: Bearing-Order Feature Extraction

## Purpose

Component 7 reduces the four spectra and the accumulated statistics to the 32 values used by the AI model.

Each accelerometer produces the same eight features:

| Local index | Feature | Meaning |
| :---: | :--- | :--- |
| 0 | Band-pass acceleration RMS | Overall vibration level in the selected resonance band |
| 1 | Band-pass acceleration kurtosis | How impulsive the vibration is |
| 2 | Band-pass acceleration crest factor | Largest impact relative to the RMS level |
| 3 | Smooth-envelope RMS | Overall strength of the bearing-impact envelope |
| 4 | FTF-family magnitude | Cage-related spectral evidence |
| 5 | BSF-family magnitude | Rolling-element-related spectral evidence |
| 6 | BPFO-family magnitude | Outer-race-related spectral evidence |
| 7 | BPFI-family magnitude | Inner-race-related spectral evidence |

The first three features come from the running statistics updated in Component 3. Envelope RMS comes from Component 5. The four bearing-family values come from the envelope spectrum produced by Component 6.

## Bearing-Order Bands

The baseline configuration uses the 6200-class bearing orders supplied by the mechanical team:

| Family | Configured order | At 1750 RPM | At 3600 RPM |
| :--- | :---: | :---: | :---: |
| FTF | 0.38 | 11.1 Hz | 22.8 Hz |
| BSF | 1.98 | 57.8 Hz | 118.8 Hz |
| BPFO | 3.05 | 89.0 Hz | 183.0 Hz |
| BPFI | 4.95 | 144.4 Hz | 297.0 Hz |

For every window, Component 7 uses the measured RPM rather than copying the nominal operating-mode value:

> Target frequency = configured order × measured RPM ÷ 60

Each family feature measures the combined magnitude of FFT bins in a narrow band around its configured fundamental order. A half-width of ±0.10 order is used initially. Component 7 converts both edges of that order band to hertz using measured RPM, then combines the spectrum bins inside it. Harmonics and sidebands are added only if measured data shows they improve classification.

## Fixed Feature Order

```text
features[0..7]   = accelerometer 1 features
features[8..15]  = accelerometer 2 features
features[16..23] = accelerometer 3 features
features[24..31] = accelerometer 4 features
```

The same ordering and feature definitions must be used by:

- STM32 firmware.
- Stored training data.
- AI training code.
- AI inference code.
- Dashboard labels.

If any required accelerometer data, Keyphasor timing, or bearing configuration is invalid, the complete window is skipped.

## Output

- One ordered array of 32 Float32 features.
- Total feature-array size: 32 × 4 bytes = 128 bytes.

---

# Component 8: Synchronous 1x Displacement Extraction

## Purpose

Component 8 handles the two radial proximity-probe signals, one at each measurement plane. It measures the shaft-displacement component that occurs once per shaft revolution, called the 1x component.

The FE digital twin needs both amplitude and phase from the two measured radial directions.

## Separate Processing Branch

The proximity channels leave Component 2 and enter Component 8 directly. They do not pass through the accelerometer band-pass, rectification, envelope, or bearing-order feature components.

## Streaming Operation

The ending Keyphasor boundary of a revolution is needed before the software knows the exact duration of that revolution. Therefore, Component 8 handles one completed revolution at a time:

1. Retain the calibrated displacement samples for the current revolution.
2. When the next Keyphasor edge arrives, map those samples to their angular positions within that revolution.
3. Remove the average probe gap so only changing shaft motion remains.
4. Update running sine and cosine totals for exactly 1 shaft order.
5. Discard that revolution's temporary displacement samples.
6. Continue accumulating through all 20 revolutions.

This avoids storing the complete raw proximity window and avoids calculating an unnecessary displacement FFT.

## Final Calculation

At `t20`, the final sine and cosine totals are converted into:

- 1x peak amplitude in micrometres.
- 1x phase in radians relative to the Keyphasor.

The output order is fixed:

```text
0: Plane 1 radial amplitude
1: Plane 1 radial phase
2: Plane 2 radial amplitude
3: Plane 2 radial phase
```

Reversing a probe's physical direction changes its phase. The probe names and positive directions must therefore be confirmed once during installation and then kept consistent with the FE model.

If the proximity acquisition or Keyphasor timing is invalid, the complete window is skipped.

## Output

- Four Float32 values: two peak amplitudes and two phases.
- Total displacement-result size: 4 × 4 bytes = 16 bytes.

---

# Component 9: Simple Payload Construction

## Purpose

Component 9 combines the result of the accelerometer branch and the proximity branch into one fixed payload.

It does not perform AI inference or FE calculations. Those operations run on the local host.

## Payload Layout

| Byte offset | Type | Field |
| :---: | :--- | :--- |
| 0–3 | Float32 | Average RPM over the 20-revolution window |
| 4–131 | Float32[32] | Ordered accelerometer features |
| 132–135 | Float32 | Plane 1 radial amplitude in micrometres peak |
| 136–139 | Float32 | Plane 1 radial phase in radians |
| 140–143 | Float32 | Plane 2 radial amplitude in micrometres peak |
| 144–147 | Float32 | Plane 2 radial phase in radians |
| 148–151 | UInt32 | `t20` timestamp in milliseconds since startup |

Payload size:

> 4 bytes RPM + 128 bytes features + 16 bytes displacement + 4 bytes timestamp = 152 bytes

## Simple Rules

- The STM32 writes the fields in the fixed order shown above.
- All Float32 and UInt32 values use little-endian byte order.
- The STM32 sends the 152 bytes to the ESP32-C6 through UART.
- The ESP32-C6 forwards the same bytes over local Wi-Fi.
- The host uses the same fixed table to decode the message.
- If acquisition or processing fails, the STM32 skips that window instead of sending invented values.

A short UART start marker can be placed before the payload if needed to locate the beginning of a message. It is not part of the 152-byte application payload.

The prototype does not require schema words, a large status bitmask, NaN encoding rules, acknowledgements, retries, or runout-compensation fields. These can be added later only if testing proves they are needed.

## Processing Deadline

The Specification 5 timer starts at `t20`, the Keyphasor edge that completes revolution 20.

It stops when the completed 152-byte payload is available in STM32 memory.

The required result is:

> Payload-ready time − `t20` must be 333 ms or less.

Most filtering and statistical accumulation already happened while the blocks were arriving. After `t20`, the remaining work mainly consists of FFTs, order mapping, final feature calculations, final displacement calculations, and writing 152 bytes.

---

## 5. End-to-End Working Example

Assume the rotor runs steadily at 3600 RPM.

### Acquisition size

- 3600 RPM equals 60 revolutions per second.
- Twenty revolutions take 20 ÷ 60 = 1/3 second, approximately 333.3 ms.
- At 30 kSamples/s, every channel produces 10,000 raw samples over this interval.
- A 256-sample DMA block fills in 8.533 ms.
- The 10,000 sample sets are equivalent to 39 full DMA blocks plus 16 sample sets: 39 × 256 + 16 = 10,000. If the window starts at a DMA-block boundary, it occupies 39 complete blocks and the first 16 sets of the next block; otherwise the boundary blocks are split according to their timestamps.

For explanation only, these real DMA blocks can be viewed as two groups:

- Group A contains the data covering revolutions 1–10.
- Group B contains the data covering revolutions 11–20.

A Keyphasor boundary can occur inside a DMA block, so the groups are defined by time and shaft revolution, not by DMA memory boundaries.

### What happens while Group A arrives

1. Component 1 acquires alternating 256-sample blocks and records Keyphasor edges.
2. Component 2 converts the ADC codes into acceleration and displacement.
3. Component 3 filters the acceleration and updates its running RMS, kurtosis, and peak information.
4. Component 4 takes the absolute value of every filtered acceleration sample.
5. Component 5 smooths and decimates the envelope, then appends the retained samples to the current window record.
6. Component 8 uses each completed revolution of proximity data to update its running 1x sine and cosine totals.

No AI feature vector or payload is produced after Group A. The window has only reached 10 revolutions.

### What happens while Group B arrives

The same operations continue without resetting the filters or running totals.

- Component 3 adds Group B to the statistics started during Group A.
- Component 5 appends Group B's reduced envelope samples after Group A's samples.
- Component 8 adds the last ten revolutions to the same displacement totals.
- Component 1 closes the window when the twentieth revolution ends at `t20`.

At the selected 5 kSamples/s envelope rate, each accelerometer now has approximately:

> 5,000 samples/s × 1/3 second ≈ 1,667 envelope samples

### What happens after `t20`

1. Component 6 removes the envelope mean, applies a Hann window, zero-pads each approximately 1,667-sample record to 4,096 values, and calculates one frequency spectrum per accelerometer.
2. Component 6 uses the measured average RPM to associate every frequency bin with a shaft order. The 20-revolution record provides 0.05-order physical resolution; zero-padding only makes the displayed FFT grid denser.
3. Component 7 combines the accumulated statistics and the FTF, BSF, BPFO, and BPFI order-band magnitudes into 8 features per channel, producing 32 features.
4. Component 8 converts its completed 1x totals into amplitude and phase for the Plane 1 and Plane 2 radial probes.
5. Component 9 combines average RPM, the 32 features, the four displacement values, and the timestamp into one 152-byte payload.
6. The STM32 sends the payload to the ESP32-C6.
7. The ESP32-C6 forwards it over local Wi-Fi to the AI and FE host.
8. The AI makes one prediction for the complete 20-revolution window.
9. The FE digital twin makes one physics update using the displacement from the same window.

At the other supported mode, 1750 RPM, the same sequence covers approximately 685.7 ms, with about 20,571 raw samples and 3,429 envelope samples per channel. Each actual envelope record is zero-padded to 4,096 values for the FFT, and the output still contains 32 acceleration features, four displacement values, and one 152-byte payload.

The DMA blocks are therefore a memory and scheduling method. The AI and FE digital twin still receive exactly one result representing all 20 revolutions.

---

## 6. Minimal Implementation Checks

The following checks are enough for the prototype:

1. Set `OS[2:0] = 000` and confirm that the AD7606 transfers all eight channel words at 30 kSample-sets/s without data loss; V1–V6 carry active signals and V7–V8 are reserved.
2. At 10 MHz SPI, confirm that every 128-bit DMA read started after `BUSY` falls completes before the following `BUSY` falling edge, and that streaming processing keeps up with the 256-sample blocks arriving every 8.533 ms.
3. Confirm that one logical window contains exactly 20 valid Keyphasor revolutions at each supported mode, 1750 RPM and 3600 RPM, and that transition windows are rejected.
4. Confirm that Component 6 uses every envelope sample in the 20-revolution window and zero-pads, rather than truncates or angle-resamples, each record to the 4,096-point FFT size.
5. Confirm that Component 7 produces exactly 32 features in the documented order.
6. Confirm that Component 8 produces exactly four displacement values in the documented order.
7. Confirm that the serialized payload is exactly 152 bytes and decodes correctly on the host.
8. Measure from `t20` until the payload is ready and confirm the time is 333 ms or less.
9. Disconnect the internet and confirm that STM32, ESP32-C6, local Wi-Fi, AI, and FE processing continue locally.

Calculations show that the design is feasible. These requirements should be described as **design complete—calculated** until the tests above are performed on the actual hardware.

---

## 7. Items That Still Require Real Hardware

No additional answer from the mechanical team is required before implementing the baseline pipeline. The following installed-system checks remain part of commissioning:

1. Enter the calibration sensitivity and offset for each installed accelerometer and proximity channel.
2. Verify the complete SRD-1104 → external fourth-order LPF → AD7606 response over 2–8 kHz, the attenuation above the intended passband, and electrical compatibility at the ADC inputs.
3. Verify reliable Keyphasor edge capture at both operating modes.
4. Start Component 3 with the selected 2–8 kHz band and refine it only if measured data identifies a stronger impact-sensitive band.
5. Verify that the selected envelope low-pass cutoff preserves the highest order feature used by the classifier before decimation to 5 kSamples/s.
6. Monitor the Keyphasor revolution periods during tests. Add angular resampling only if meaningful within-window speed change produces visible order smearing or unstable spectral features.

These checks tune or verify the installed system; they do not leave the pipeline undefined. The nine-component software structure, 256-sample DMA block, 20-revolution window, 4,096-point zero-padded FFT, supplied bearing-order centres, 32-feature order, four displacement values, and 152-byte payload are defined.
