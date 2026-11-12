"""Export independent time-feature and envelope references for the current pipeline.

Continuously repeat the existing float32 raw snippets across two nominal windows.
No filter or decimation state is reset between those windows.
"""
import argparse
from pathlib import Path
import hashlib
import json
import re
import sys

import numpy as np
from scipy.signal import sosfilt


def literal(value):
    """Format a float32 value as a reproducible C floating-point literal."""
    return f"{float(np.float32(value)):.9e}f"


def rows(values):
    """Format expected sample arrays as six C literals per line."""
    return "\n".join("        " + ", ".join(literal(v) for v in values[i:i+6]) + ","
                     for i in range(0, len(values), 6))


def main():
    """Export complete-window references at both nominal speeds and record their source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-project", type=Path, required=True)
    parser.add_argument("--firmware-project", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.python_project))
    from steps.step_02.step_02_bandpass import design_bandpass, apply_bandpass
    from steps.step_03.step_03_time_domain_features import extract_time_domain_features
    from steps.step_04.step_04_envelope import design_envelope_lowpass, apply_envelope_lowpass, rectify_signal
    from steps.step_05.step_05_decimation import decimate_envelope

    data = (args.firmware_project / "CM7/TestData/fixtures/bandpass_fixtures.h").read_bytes()
    match = re.search(r"bandpass_test_input\[4\]\[1024\] = \{(.*?)\n\};", data.decode(), re.S)
    if match is None:
        raise ValueError("Could not locate existing four-channel input fixtures.")
    values = re.findall(r"([-+]?\d+\.\d+e[-+]\d+)f", match.group(1))
    if len(values) != 4096:
        raise ValueError("Expected exactly 4 x 1024 float literals.")
    snippets = np.array(values, dtype=np.float32).reshape(4, 1024)
    bandpass = design_bandpass(30000, 2000, 8000)
    lowpass = design_envelope_lowpass(30000, 350, 1250)
    expected, envelopes, counts, offsets, quantized_errors = [], [], [], [], []
    for rpm in (1750, 3600):
        count = round(30000 * 60 * 20 / rpm)
        per_channel = []
        envelope_windows = [[], []]
        errors = []
        for channel in snippets:
            replay = np.tile(channel, (2 * count + 1023) // 1024)[:2*count]
            filtered, _ = apply_bandpass(replay.astype(np.float64), bandpass)
            per_channel.append([extract_time_domain_features(filtered[w*count:(w+1)*count]).astype(np.float32)
                                for w in range(2)])
            envelope, _ = apply_envelope_lowpass(rectify_signal(filtered), lowpass)
            filtered32 = sosfilt(bandpass.astype(np.float32), replay)
            envelope32 = sosfilt(lowpass.astype(np.float32), rectify_signal(filtered32))
            error = np.abs(envelope32.astype(np.float64) - envelope)
            # This new cascade has rounding from both IIR filters. Preserve the
            # existing 2e-5 standalone/feature checks and budget 4e-5 here.
            if np.any(error > 4e-5 * (1 + np.abs(envelope))):
                raise ValueError("The quantized cascade exceeds its envelope sample tolerance.")
            errors.append(float(np.max(error)))
            phase = 0
            for window in range(2):
                reduced, phase = decimate_envelope(envelope[window*count:(window+1)*count], 6, phase)
                envelope_windows[window].append(reduced.astype(np.float32))
                if phase != ((window + 1) * count) % 6:
                    raise ValueError("Unexpected decimation phase.")
        expected.append(np.array(per_channel).transpose(1, 0, 2))
        envelopes.append(envelope_windows)
        counts.append([len(w[0]) for w in envelope_windows])
        offsets.append([0, (-count) % 6])
        quantized_errors.append(errors)

    # Preserve the preexisting time-feature oracle's values and serialization.
    lines = ["/* Generated Python oracle: continuously repeated CWRU snippets. */",
             "/* Functional test input, not actual 3600 RPM machine measurements. */",
             "#ifndef COE_PIPELINE_EXPECTED_H", "#define COE_PIPELINE_EXPECTED_H",
             "static const TimeDomainFeatures pipeline_expected[2][2][4] = {"]
    for mode in expected:
        lines += ["    {"]
        for window in mode:
            lines += ["        {"]
            for features in window:
                lines += ["            {" + ", ".join(f"{float(v):.9e}f" for v in features) + "},"]
            lines += ["        },"]
        lines += ["    },"]
    lines += ["};", "#endif", ""]
    target = args.firmware_project / "CM7/TestData/fixtures"
    (target / "pipeline_expected.h").write_text("\n".join(lines), encoding="utf-8")

    header = ["/* Python steps 02/04/05: continuously filtered raw float32 input snippets. */",
              "/* Complete reduced envelopes; no state resets between windows. */",
              "#ifndef COE_PIPELINE_ENVELOPE_EXPECTED_H", "#define COE_PIPELINE_ENVELOPE_EXPECTED_H"]
    for mode, rpm in enumerate((1750, 3600)):
        for window in range(2):
            header.append(f"static const float pipeline_envelope_{rpm}_{window}[4][{counts[mode][window]}] = {{")
            for channel in envelopes[mode][window]:
                header += ["    {", rows(channel), "    },"]
            header.append("};")
    for name, values in (("pipeline_envelope_counts", counts), ("pipeline_envelope_first_offsets", offsets)):
        header += [f"static const uint32_t {name}[2][2] = {{"]
        for mode in values:
            header.append("    {" + ", ".join(f"{v}U" for v in mode) + "},")
        header.append("};")
    header.append("static const float *const pipeline_envelope_expected[2][2][4] = {")
    for rpm in (1750, 3600):
        header.append("    {")
        for window in range(2):
            header.append("        {" + ", ".join(f"pipeline_envelope_{rpm}_{window}[{c}]" for c in range(4)) + "},")
        header.append("    },")
    header += ["};", "#endif", ""]
    envelope_text = "\n".join(header)
    (target / "pipeline_envelope_expected.h").write_text(envelope_text, encoding="utf-8")

    report = {
        "source_fixture_sha256": hashlib.sha256(data).hexdigest(),
        "input": "repeat each float32 1024-sample snippet continuously across two windows",
        "sample_rate_hz": 30000, "revolutions": 20,
        "rpm_modes": [1750, 3600], "samples_per_window": [20571, 10000],
        "filter_initial_state": "zero only at stream startup",
        "reference": "unchanged Python steps 02/03/04/05, float64 calculations",
        "tolerance": "abs(error) <= 2e-5 * (1 + abs(expected))",
        "envelope_tolerance": "abs(error) <= 4e-5 * (1 + abs(expected))",
        "envelope_tolerance_reason": "combined float32 band-pass and low-pass rounding; standalone/feature tests unchanged",
        "features": np.asarray(expected).tolist(),
        "envelope_sample_rate_hz": 5000, "envelope_counts": counts,
        "envelope_first_input_offsets": offsets,
        "max_scipy_float32_cascade_error_g": quantized_errors,
        "envelope_header_sha256": hashlib.sha256(envelope_text.encode()).hexdigest(),
        "mcu_execution": "pending integrated board verification",
    }
    (target / "pipeline_reference_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Generated time features and every envelope sample for two modes/two consecutive windows/four channels.")
    print("Envelope counts:", counts, "First local indices:", offsets)
    print("Maximum SciPy float32 cascade errors:", quantized_errors)


if __name__ == "__main__":
    main()
