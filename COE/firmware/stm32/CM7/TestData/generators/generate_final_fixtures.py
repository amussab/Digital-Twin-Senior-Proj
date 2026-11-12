"""Export independent Python FFT, frequency-feature and wire-packet references.

Uses the unchanged Python steps 02..08 on exactly specified Float32 inputs.
Synthetic records exercise FFT endpoints, odd lengths, padding and zero/DC cases.
Full-pipeline vectors cover two bearing profiles, two speeds and three windows.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import struct
import sys
import zlib
import numpy as np
from scipy.signal import sosfilt


def literal(value):
    """Format one Float32 number as a C literal."""
    return f'{float(np.float32(value)):.9e}f'


def c_rows(values):
    """Format a flat numeric array six values per line."""
    return '\n'.join('    ' + ', '.join(literal(v) for v in values[i:i+6]) + ','
                     for i in range(0, len(values), 6))


def main():
    """Write reference headers and a report; do not alter prior component fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python-project', type=Path, required=True)
    parser.add_argument('--firmware-project', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.python_project))
    from steps.step_02.step_02_bandpass import design_bandpass, apply_bandpass
    from steps.step_03.step_03_time_domain_features import extract_time_domain_features
    from steps.step_04.step_04_envelope import design_envelope_lowpass, rectify_signal, apply_envelope_lowpass
    from steps.step_05.step_05_decimation import decimate_envelope
    from steps.step_06.step_06_envelope_fft import calculate_envelope_spectrum
    from steps.step_07.step_07_frequency_domain_features import (
        calculate_envelope_rms, extract_frequency_domain_features,
        CWRU_DRIVE_END_ORDERS, PROJECT_BEARING_ORDERS, FAMILIES,
    )
    from steps.step_08.step_08_feature_vector import build_feature_vector
    target = args.firmware_project / "CM7/TestData/fixtures"

    generator = np.random.default_rng(74551)
    cases = []
    for count in (3429, 3428, 1667, 1666):
        t = np.arange(count) / 5000
        cases.append((0.7 + 0.2*np.sin(2*np.pi*31.7*t)
                      + 0.1*np.cos(2*np.pi*143.5*t)
                      + generator.normal(0, 0.015, count)).astype(np.float32))
    cases += [np.array([0.1, 0.8, 1.3], dtype=np.float32),
              np.zeros(1024, dtype=np.float32), np.full(1024, 0.8, dtype=np.float32),
              (1 + 0.5*(-1.0)**np.arange(2048)).astype(np.float32),
              (1 + 0.4*np.cos(2*np.pi*100*np.arange(2048)/4096)).astype(np.float32)]
    header = ['/* Python step 06/07: every spectral bin for deterministic Float32 inputs. */',
              '#ifndef COE_FINAL_SPECTRUM_FIXTURES_H', '#define COE_FINAL_SPECTRUM_FIXTURES_H']
    synthetic_features = []
    for index, record in enumerate(cases):
        frequencies, amplitude, density = calculate_envelope_spectrum(record, 5000, 4096)
        # Independent conservation check for the reference's one-sided PSD scale.
        taper = 0.5 - 0.5*np.cos(2*np.pi*np.arange(len(record))/len(record))
        weighted_power = np.sum(((record.astype(float)-np.mean(record.astype(float)))*taper)**2) / np.sum(taper**2)
        if not np.isclose(np.sum(density)*5000/4096, weighted_power, rtol=1e-12, atol=1e-14):
            raise ValueError('Reference PSD failed the Parseval power check.')
        if index == 7 and not np.isclose(amplitude[-1], 0.5, atol=1e-12):
            raise ValueError('Nyquist endpoint normalization is incorrect.')
        if index == 8 and not np.isclose(amplitude[100], 0.4, atol=1e-7):
            raise ValueError('Known-tone peak amplitude is incorrect.')
        for suffix, data in (('input', record), ('amplitude', amplitude), ('density', density)):
            header += [f'static const float final_fft_{suffix}_{index}[{len(data)}] = {{', c_rows(data), '};']
        synthetic_features.append([
            [calculate_envelope_rms(record), *extract_frequency_domain_features(
                frequencies, density, rpm, CWRU_DRIVE_END_ORDERS)]
            for rpm in (1750, 3600)
        ])
    for suffix in ('input', 'amplitude', 'density'):
        header += [f'static const float *const final_fft_{suffix}[9] = {{',
                   '    ' + ', '.join(f'final_fft_{suffix}_{i}' for i in range(9)) + ',', '};']
    header += ['static const uint32_t final_fft_counts[9] = {3429U, 3428U, 1667U, 1666U, 3U, 1024U, 1024U, 2048U, 2048U};',
               'static const FrequencyDomainFeatures final_frequency_expected[9][2] = {']
    for mode_features in synthetic_features:
        header += ['    {']
        for features in mode_features:
            header += ['        {' + literal(features[0]) + ', {'
                       + ', '.join(literal(v) for v in features[1:]) + '}},']
        header += ['    },']
    header += ['};', '#endif', '']
    (target/'final_spectrum_fixtures.h').write_text('\n'.join(header), encoding='utf-8', newline='\n')

    source = (args.firmware_project/'CM7/TestData/fixtures/bandpass_fixtures.h').read_bytes()
    match = re.search(r'bandpass_test_input\[4\]\[1024\] = \{(.*?)\n\};', source.decode(), re.S)
    if match is None:
        raise ValueError('Original replay snippets missing.')
    values = re.findall(r'([-+]?\d+\.\d+e[-+]\d+)f', match[1])
    snippets = np.array(values, dtype=np.float32).reshape(4, 1024)
    bp = design_bandpass(30000, 2000, 8000)
    lp = design_envelope_lowpass(30000, 350, 1250)
    vectors = np.empty((2, 2, 3, 32), dtype=np.float32)
    third_envelopes = [[], []]
    cascade_errors = []
    for mode, rpm in enumerate((1750, 3600)):
        count = round(30000*60*20/rpm)
        for channel in range(4):
            replay = np.tile(snippets[channel], (3*count+1023)//1024)[:3*count]
            filtered, _ = apply_bandpass(replay.astype(float), bp)
            smooth, _ = apply_envelope_lowpass(rectify_signal(filtered), lp)
            quantized = sosfilt(lp.astype(np.float32), np.abs(sosfilt(bp.astype(np.float32), replay)))
            phase = 0
            for window in range(3):
                envelope, phase = decimate_envelope(smooth[window*count:(window+1)*count], 6, phase)
                if window == 2:
                    third_envelopes[mode].append(envelope.astype(np.float32))
                frequency, _, density = calculate_envelope_spectrum(envelope, 5000, 4096)
                original_time = extract_time_domain_features(filtered[window*count:(window+1)*count])
                for profile, orders in enumerate((CWRU_DRIVE_END_ORDERS, PROJECT_BEARING_ORDERS)):
                    original_frequency = [calculate_envelope_rms(envelope),
                        *extract_frequency_domain_features(frequency, density, rpm, orders)]
                    vectors[profile, mode, window, channel*8:(channel+1)*8] = build_feature_vector(
                        original_time, original_frequency)
                    # Profile coefficient quantization must not change selected bin centers.
                    for name in FAMILIES:
                        edges = ((orders[name]-0.1)*rpm/60, (orders[name]+0.1)*rpm/60)
                        edges32 = ((float(np.float32(orders[name]))-float(np.float32(0.1)))*rpm/60,
                                   (float(np.float32(orders[name]))+float(np.float32(0.1)))*rpm/60)
                        selected = (frequency >= edges[0]) & (frequency <= edges[1])
                        if not np.array_equal(selected, (frequency >= edges32[0]) & (frequency <= edges32[1])):
                            raise ValueError('Quantized bearing orders changed a selected feature bin.')
                    first = (-(window*count)) % 6
                    reduced32 = quantized[window*count+first:(window+1)*count:6]
                    f32, _, density32 = calculate_envelope_spectrum(reduced32, 5000, 4096)
                    approximate = np.array([calculate_envelope_rms(reduced32),
                        *extract_frequency_domain_features(f32, density32, rpm, orders)])
                    error = np.max(np.abs(approximate - original_frequency))
                    cascade_errors.append(float(error))
                    if np.any(np.abs(approximate-original_frequency) > 4e-5*(1+np.abs(original_frequency))):
                        raise ValueError('Float32 cascade exceeds the new frequency-feature tolerance.')
    header = ['/* Unchanged Python steps 02..08: two profiles/modes, three windows; four channels. */',
              '#ifndef COE_FINAL_PIPELINE_EXPECTED_H', '#define COE_FINAL_PIPELINE_EXPECTED_H',
              'static const float final_pipeline_expected[2][2][3][32] = {']
    for profile in vectors:
        header += ['    {']
        for mode in profile:
            header += ['        {']
            for vector in mode:
                header += ['            {', c_rows(vector), '            },']
            header += ['        },']
        header += ['    },']
    header += ['};', '#endif', '']
    (target/'final_pipeline_expected.h').write_text('\n'.join(header), encoding='utf-8', newline='\n')

    header = ['/* Additional third-window envelope oracle; previous two-window fixtures are unchanged. */',
              '#ifndef COE_FINAL_THIRD_ENVELOPES_H', '#define COE_FINAL_THIRD_ENVELOPES_H']
    for mode, rpm in enumerate((1750, 3600)):
        header += [f'static const float final_third_envelope_{rpm}[4][{len(third_envelopes[mode][0])}] = {{']
        for record in third_envelopes[mode]:
            header += ['    {', c_rows(record), '    },']
        header += ['};']
    header += ['static const uint32_t final_third_envelope_counts[2] = {3429U, 1666U};',
               'static const uint32_t final_third_envelope_offsets[2] = {0U, 4U};',
               'static const float *const final_third_envelopes[2][4] = {']
    for rpm in (1750, 3600):
        header += ['    {'+', '.join(f'final_third_envelope_{rpm}[{c}]' for c in range(4))+'},']
    header += ['};', '#endif', '']
    (target/'final_third_envelopes.h').write_text('\n'.join(header), encoding='utf-8', newline='\n')

    # The four production record lengths have immutable Hann coefficients in Flash.
    # Avoid computing thousands of cosine values inside a DMA window-closing call.
    header = ['/* Generated periodic Hann Float32 weights; normalization uses actual weight sums. */',
              '#ifndef COE_ENVELOPE_TAPERS_H', '#define COE_ENVELOPE_TAPERS_H']
    for count in (3429, 3428, 1667, 1666):
        weights = (0.5-0.5*np.cos(2*np.pi*np.arange(count)/count)).astype(np.float32)
        header += [f'static const float envelope_taper_{count}[{count}] = {{', c_rows(weights), '};']
    header += ['#endif', '']
    (args.firmware_project/'CM7/App/dsp/envelope_spectrum/envelope_tapers.h').write_text('\n'.join(header), encoding='utf-8', newline='\n')

    # Exact distinct synthetic values make ordering, float bits and CRC testable byte-for-byte.
    packet_values = np.array([[c*10+1, c*10+2, c*10+3, c*10+4,
                               c*10+5, c*10+6, c*10+7, c*10+8] for c in range(4)], dtype=np.float32).ravel()
    packet = struct.pack('<4sHHIIIII', b'COE1', 1, 32, 17, 1750, 1, 15, 1) + packet_values.astype('<f4').tobytes()
    packet += struct.pack('<I', zlib.crc32(packet))
    if len(packet) != 160:
        raise ValueError('Payload reference has incorrect size.')
    header = ['/* Python struct.pack + zlib.crc32, independent wire-format oracle. */',
              '#ifndef COE_PAYLOAD_FIXTURE_H', '#define COE_PAYLOAD_FIXTURE_H',
              'static const uint8_t final_payload_expected[160] = {']
    for offset in range(0, len(packet), 16):
        header += ['    ' + ', '.join(f'0x{b:02X}U' for b in packet[offset:offset+16]) + ',']
    header += ['};', '#endif', '']
    (target/'final_payload_fixture.h').write_text('\n'.join(header), encoding='utf-8', newline='\n')
    (target/'final_payload_reference.bin').write_bytes(packet)
    report = {
        'source_fixture_sha256': hashlib.sha256(source).hexdigest(),
        'reference': 'unchanged Python steps 02 through 08, Float64 DSP and Float32 vector conversion',
        'fft_size': 4096, 'fft_bins': 2049, 'bin_spacing_hz': 5000/4096,
        'synthetic_fft_records': len(cases), 'profiles': ['CWRU drive end', 'provisional project'],
        'rpm_modes': [1750, 3600], 'channels': 4, 'feature_count': 32,
        'complete_pipeline_reference_windows': 12,
        'vectors': vectors.tolist(),
        'time_feature_tolerance': '2e-5 * (1+abs(reference)), preserved',
        'new_spectrum_tolerance': '2e-5 * (1+abs(reference))',
        'new_frequency_feature_tolerance': '4e-5 * (1+abs(reference))',
        'max_scipy_quantized_frequency_feature_error_g': max(cascade_errors),
        'payload_bytes': 160, 'payload_crc32_hex': hex(zlib.crc32(packet[:-4])),
        'parseval_check': 'passed on all nine synthetic records',
        'nyquist_endpoint_and_known_tone_checks': 'passed',
        'bearing_band_bin_selection_after_float32_quantization': 'unchanged at both modes/profiles',
        'hardware': 'reference export does not execute the MCU; see performance.md for board results',
    }
    (target/'final_reference_report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'vectors'}, indent=2))


if __name__ == '__main__':
    main()
