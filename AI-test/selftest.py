"""Self-checks that need no trained model, no downloads and no rig.

These exist because several numbers this project depends on were supplied by
one discipline and consumed by another without anything in between ever
checking them. Each test here is cheap, deterministic, and fails loudly.
"""

from __future__ import annotations

import numpy as np

import dsp
import payload
from config import BEARING_ORDERS, FEATURE_NAMES, N_FEATURES


def check_payload_codec() -> tuple[bool, str]:
    """152-byte wire format round-trips exactly [COE Component 9]."""
    rng = np.random.default_rng(0)
    window = payload.Window(
        rpm=1750.4,
        features=rng.normal(1.0, 0.3, N_FEATURES).astype(np.float32),
        p1_amp_um=12.5, p1_phase_rad=0.87,
        p2_amp_um=9.25, p2_phase_rad=-1.44,
        t20_ms=123_456_789,
    )
    raw = payload.encode(window)
    if len(raw) != 152:
        return False, f"payload is {len(raw)} bytes, expected 152"
    back = payload.decode(raw)
    if not np.allclose(back.features, window.features, atol=1e-6):
        return False, "feature vector did not survive the round trip"
    if back.t20_ms != window.t20_ms:
        return False, "timestamp did not survive the round trip"
    try:
        payload.decode(raw + b"\x00" * 16)      # the FDR's 168-byte claim
        return False, "a 168-byte payload was accepted; it must be rejected"
    except ValueError:
        pass
    return True, "152-byte round trip exact; wrong-length payloads rejected"


def check_bearing_orders() -> tuple[bool, str]:
    """COE's supplied bearing orders are internally consistent.

    Independent check of numbers supplied by the mechanical team and used
    directly by Component 7's order bands. Two kinematic identities must hold
    for any real bearing: BPFO + BPFI = N (the rolling-element count), and
    FTF = BPFO / N. If they do not, at least one supplied order is wrong and
    the order bands are integrating the wrong frequencies.
    """
    implied_n = BEARING_ORDERS["BPFO"] + BEARING_ORDERS["BPFI"]
    n = round(implied_n)
    if abs(implied_n - n) > 0.02:
        return False, f"BPFO+BPFI = {implied_n:.3f}, not a whole element count"

    ratio = 1.0 - 2.0 * BEARING_ORDERS["BPFO"] / n
    geometry = dsp.BearingGeometry(
        n_rolling_elements=n,
        ball_diameter_mm=ratio * 30.0,
        pitch_diameter_mm=30.0,
        name="implied from supplied orders",
    )
    derived = dsp.bearing_orders_from_geometry(geometry)
    worst = max(abs(derived[k] - BEARING_ORDERS[k]) for k in BEARING_ORDERS)
    if worst > 0.02:
        return False, f"supplied orders disagree with geometry by {worst:.3f} order"
    return True, (
        f"consistent with an {n}-element bearing, d/D = {ratio:.4f}; "
        f"all four orders reproduce to within {worst:.4f} order"
    )


def check_feature_pipeline() -> tuple[bool, str]:
    """The DSP chain recovers a seeded fault at the right order.

    Synthesise impacts at a known bearing order, run COE Components 3-7, and
    require that the corresponding order-family feature dominates. This is what
    makes the 32-feature vector meaningful; if it fails, no amount of model
    training recovers the fault type.
    """
    sample_rate, rpm = 25_600.0, 1750.0
    cfg = dsp.PipelineConfig(sample_rate_hz=sample_rate)
    n = dsp.samples_per_window(rpm, cfg)
    t = np.arange(n) / sample_rate
    rng = np.random.default_rng(3)
    families = ["FTF", "BSF", "BPFO", "BPFI"]
    results = []

    for injected in ("BPFO", "BPFI", "BSF", None):
        sig = 0.05 * rng.standard_normal(n) + 0.3 * np.sin(2 * np.pi * (rpm / 60) * t)
        if injected:
            impact_hz = BEARING_ORDERS[injected] * rpm / 60.0
            for k in range(int(t[-1] * impact_hz) + 1):
                t0 = k / impact_hz
                mask = (t >= t0) & (t < t0 + 0.004)
                sig[mask] += 1.4 * np.exp(-900 * (t[mask] - t0)) * np.sin(
                    2 * np.pi * 4200 * (t[mask] - t0)
                )
        features = dsp.channel_features(sig, rpm, cfg)
        magnitudes = dict(zip(families, features[4:8]))
        dominant = max(magnitudes, key=magnitudes.get)
        share = magnitudes[dominant] / (sum(magnitudes.values()) + 1e-12)

        if injected is None:
            # Healthy: no family should dominate, and kurtosis should sit near
            # the Gaussian value of 3.
            if share > 0.55 or not (2.0 < features[1] < 4.5):
                results.append(f"healthy case looked faulted (share {share:.2f}, "
                               f"kurtosis {features[1]:.2f})")
        elif dominant != injected:
            results.append(f"{injected} impacts read as {dominant}")
        elif share < 0.60:
            results.append(f"{injected} recovered but only {share:.2f} of band energy")

    if results:
        return False, "; ".join(results)
    return True, "seeded BPFO/BPFI/BSF impacts each recovered at the right order"


def check_feature_contract() -> tuple[bool, str]:
    """The 32-feature vector matches COE's declared layout."""
    if len(FEATURE_NAMES) != N_FEATURES:
        return False, f"{len(FEATURE_NAMES)} names for {N_FEATURES} features"
    expected = ["a1_bp_rms", "a2_bp_rms", "a3_bp_rms", "a4_bp_rms"]
    if [FEATURE_NAMES[i * 8] for i in range(4)] != expected:
        return False, "feature blocks are not grouped 8-per-accelerometer"
    return True, f"{N_FEATURES} features, 8 per accelerometer, order as specified"


CHECKS = [
    ("payload codec (COE C9)", check_payload_codec),
    ("feature contract (COE C7)", check_feature_contract),
    ("bearing orders self-consistency", check_bearing_orders),
    ("DSP fault recovery (COE C3-C7)", check_feature_pipeline),
]


def run() -> int:
    print("=" * 74)
    print("ICS AI testbench -- self-checks")
    print("=" * 74)
    failures = 0
    for name, check in CHECKS:
        try:
            ok, detail = check()
        except Exception as exc:                            # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        status = "PASS" if ok else "FAIL"
        failures += 0 if ok else 1
        print(f"  {status}  {name}")
        print(f"        {detail}")
    print()
    print(f"{len(CHECKS) - failures}/{len(CHECKS)} checks passed.")
    return 1 if failures else 0
