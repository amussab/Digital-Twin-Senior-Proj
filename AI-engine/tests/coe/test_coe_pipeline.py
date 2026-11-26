"""Tests for the COE pipeline (aiengine.coe). Everything here is [SYNTHETIC]/[CALCULATED]."""
import numpy as np
import pytest
from scipy import signal

from aiengine import payload
from aiengine.coe import (CoeConfig, CoePipeline, DropInjector, FileReplaySource, HardwareSource,
                          SampleBlock, SampleSource, SimulatedRigSource, record_blocks)
from aiengine.coe.config import G


def run(src, n=2, **kw):
    out = []
    for w in CoePipeline(src, **kw).windows():
        out.append(w)
        if len(out) >= n:
            break
    return out


def sim(fault="healthy", sev=0.0, rpm=1800.0, seed=3, secs=3.0, **kw):
    return SimulatedRigSource(fault, sev, rpm, seed=seed, n_blocks=int(secs * 30000 / 256), **kw)


# ---- Component 3 / 5 filters ------------------------------------------------------------------
def test_bandpass_gain_in_and_out_of_band():
    p = CoePipeline(sim())
    f = np.array([500.0, 5000.0, 12000.0])
    _, h = signal.sosfreqz(p._sos_bp, worN=f, fs=30000.0)
    g = 20 * np.log10(np.abs(h))
    assert abs(g[1]) < 1.0            # passband ~0 dB
    assert g[0] < -40.0               # well below 2 kHz
    assert g[2] < -25.0               # above 8 kHz
    _, h2 = signal.sosfreqz(p._sos_bp, worN=[2000.0, 8000.0], fs=30000.0)
    assert np.allclose(20 * np.log10(np.abs(h2)), -3.0, atol=0.3)   # -3 dB corners


def test_envelope_lowpass_before_decimation():
    p = CoePipeline(sim())
    # node firmware: Butterworth order 7, 0.1 dB to 350 Hz, >= 60 dB from 1250 Hz (at 30 kS/s)
    assert p._sos_lp.shape[0] == 4                      # order 7 = 3 biquads + 1 first-order section
    f = np.array([50.0, 350.0, 1250.0, 5000.0])
    _, h = signal.sosfreqz(p._sos_lp, worN=f, fs=30000.0)
    g = 20 * np.log10(np.abs(h))
    assert g[0] > -0.1 and g[1] > -0.1001 and g[2] <= -60.0 + 1e-3 and g[3] < -60.0


# ---- envelope recovers a known modulation frequency -----------------------------------------------
class AmSource(SampleSource):
    """5 kHz carrier, 100 Hz sinusoidal AM, Keyphasor exactly 1800 rpm."""

    def __init__(self, n_blocks=300):
        self.cfg, self.n_blocks = CoeConfig(), n_blocks
        self.cfg.orders = dict(self.cfg.orders, BPFO=100.0 / (1800 / 60))  # put the AM line in the BPFO band

    def blocks(self):
        fs, nb = 30000.0, 256
        T = 60.0 / 1800.0
        for s in range(self.n_blocks):
            t = (s * nb + np.arange(nb)) / fs
            x = (1 + 0.8 * np.cos(2 * np.pi * 100 * t)) * np.sin(2 * np.pi * 5000 * t) * 5.0   # m/s2
            codes = np.zeros((nb, 8), dtype=np.int16)
            v = x / G * 0.1
            codes[:, 0:4] = np.rint(v[:, None] * 3276.8)
            k0, k1 = int(np.floor(s * nb / fs / T)) + 1, int(np.floor((s + 1) * nb / fs / T))
            yield SampleBlock(s, codes, np.array([k * T for k in range(k0, k1 + 1)]))


def test_envelope_recovers_modulation_frequency():
    w = run(AmSource(), 1)[0]
    f = w.features.reshape(4, 8)[0]
    assert f[6] > 10 * max(f[4], f[5], f[7])      # BPFO band holds the 100 Hz AM line


# ---- defect orders land in the right band -----------------------------------------------------------
@pytest.mark.parametrize("fault,idx", [("outer_race", 6), ("inner_race", 7), ("ball", 5)])
@pytest.mark.parametrize("rpm", [1800.0, 3600.0])
def test_defect_family_dominates(fault, idx, rpm):
    w = run(sim(fault, 0.7, rpm), 1)[0]
    fam = w.features.reshape(4, 8)[0, 4:]
    assert int(np.argmax(fam)) == idx - 4
    assert fam[idx - 4] > 2 * np.sort(fam)[-2]


def test_cage_raises_ftf_over_healthy():
    h = run(sim("healthy"), 1)[0].features.reshape(4, 8)[0, 4]
    c = run(sim("cage", 0.9), 1)[0].features.reshape(4, 8)[0, 4]
    assert c > 2 * h


def test_faulted_bearing_louder_than_other():
    f = run(sim("outer_race", 0.7), 1)[0].features.reshape(4, 8)
    assert f[0, 3] > f[2, 3] and f[0, 0] > f[2, 0]


# ---- feature layout / payload -----------------------------------------------------------------------
def test_payload_round_trip_matches_payload_module():
    w = run(sim("inner_race", 0.5), 1)[0]
    assert len(w.payload) == payload.PAYLOAD_SIZE == 152
    d = payload.decode(w.payload)
    assert np.array_equal(d.features, w.features)
    assert d.rpm == pytest.approx(w.rpm, rel=1e-6)
    assert d.t20_ms == w.t20_ms
    assert np.allclose(d.displacements, w.displacement)
    assert payload.is_valid(d)[0]


def test_displacement_recovers_injected_orbit():
    w = run(sim(), 1)[0]
    assert w.displacement[0] == pytest.approx(25.0, rel=0.02)
    assert w.displacement[1] == pytest.approx(0.6, abs=0.05)
    assert w.displacement[2] == pytest.approx(20.0, rel=0.02)
    assert w.displacement[3] == pytest.approx(1.1, abs=0.05)


def test_calibration_code_to_g():
    p = CoePipeline(sim())
    codes = np.zeros((256, 8), dtype=np.int16)
    codes[:, 0] = 3277                      # ~1.0 V on the +-10 V range = 10 g at 100 mV/g
    acc, _ = p.calibrate(codes)
    assert acc[0, 0] == pytest.approx(10.0, rel=1e-3)          # the node sends g, not m/s2
    acc5, _ = CoePipeline(sim(), CoeConfig(adc_range_v=5.0)).calibrate(codes)
    assert acc5[0, 0] == pytest.approx(5.0, rel=1e-3)   # same code = half the voltage on +-5 V


def test_features_agree_loosely_with_offline_dsp_reference():
    from aiengine.dsp import PipelineConfig, channel_features
    src = sim("outer_race", 0.7, secs=2.0)
    pipe = CoePipeline(src)
    raw = [pipe.calibrate(b.codes)[0] for b in src.blocks()]
    w = run(sim("outer_race", 0.7, secs=2.0), 1)[0]
    x = np.concatenate(raw)[:, 0]
    n0, n1 = int(round(w.t0_s * 30000)), int(round(w.t20_s * 30000))
    ref = channel_features(x[n0:n1], w.rpm, PipelineConfig(sample_rate_hz=30000.0))
    mine = w.features[:8]
    assert mine[0] == pytest.approx(ref[0], rel=0.15)           # bp_rms
    assert int(np.argmax(mine[4:])) == int(np.argmax(ref[4:]))  # same dominant family


# ---- Component 1 window discard logic ---------------------------------------------------------------
def test_window_has_20_revolutions():
    for rpm in (1800.0, 3600.0):
        w = run(sim(rpm=rpm), 1)[0]
        assert (w.t20_s - w.t0_s) == pytest.approx(1200.0 / rpm, rel=0.01)


def test_dropped_block_discards_window():
    inj = DropInjector(sim(secs=6.0), p_block_drop=0.01, seed=5)
    pipe = CoePipeline(inj, include_discarded=True)
    res = list(pipe.windows())
    assert inj.dropped_blocks > 0
    assert pipe.stats.discarded.get("dropped_block", 0) >= 1
    assert all(r.reason == "" for r in res if r.valid)


def test_out_of_range_speed_discarded_and_range_policy_accepts():
    pipe = CoePipeline(sim(rpm=2500.0), include_discarded=True)
    r = next(iter(pipe.windows()))
    assert (not r.valid) and r.reason == "rpm_out_of_range"
    cfg = CoeConfig(speed_policy="range")
    w = run(SimulatedRigSource("healthy", 0, 2500.0, seed=1, n_blocks=400, cfg=cfg), 1)[0]
    assert w.valid and w.features.shape == (32,)


def test_missed_keyphasor_edge_discards_window():
    inj = DropInjector(sim(secs=6.0), p_edge_miss=0.05, seed=2)
    pipe = CoePipeline(inj, include_discarded=True)
    list(pipe.windows())
    assert inj.missed_edges > 0
    assert pipe.stats.total_discarded >= 1


def test_speed_ramp_transition_window_rejected():
    ramp = lambda t: 1800.0 + 400.0 * t        # noqa: E731  accelerating
    pipe = CoePipeline(SimulatedRigSource("healthy", 0, 1800.0, seed=1, n_blocks=300, rpm_schedule=ramp),
                       include_discarded=True)
    res = list(pipe.windows())
    assert res and not any(r.valid for r in res)


# ---- determinism / replay / stub --------------------------------------------------------------------
def test_determinism_with_seed():
    a = [w.payload for w in run(sim("ball", 0.6, seed=9), 2)]
    b = [w.payload for w in run(sim("ball", 0.6, seed=9), 2)]
    c = [w.payload for w in run(sim("ball", 0.6, seed=10), 2)]
    assert a == b and a != c


def test_file_replay_matches_live(tmp_path):
    p = record_blocks(sim("outer_race", 0.5, seed=4), tmp_path / "rec.npz", 300)
    live = [w.payload for w in CoePipeline(sim("outer_race", 0.5, seed=4, secs=300 * 256 / 30000)).windows()]
    rep = [w.payload for w in CoePipeline(FileReplaySource(p)).windows()]
    assert live and live == rep


def test_hardware_stub_is_explicit():
    with pytest.raises(NotImplementedError):
        next(HardwareSource().blocks())


def test_udp_sender_delivers_one_152_byte_datagram_per_window():
    import socket
    from aiengine.coe.udp_sender import UdpSender, stream
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2.0)
    tx = UdpSender("127.0.0.1", rx.getsockname()[1])
    n = stream(CoePipeline(sim("outer_race", 0.5)).windows(), tx, rate_hz=0.0, max_windows=2)
    got = [rx.recvfrom(4096)[0] for _ in range(n)]
    assert n == 2 and all(len(g) == 152 for g in got)
    assert payload.decode(got[0]).rpm == pytest.approx(1800, rel=0.01)
    tx.close(); rx.close()


# ---- per-window features == hand-written reference of the node firmware formulas -------------------------
# Firmware tables/formulas: COE/firmware/stm32/CM7/App/dsp/{envelope,envelope_spectrum,frequency_features,time_features}.
ENV_LP_FIRMWARE_CMSIS = np.array([   # envelope_coefficients.h: b0,b1,b2,-a1,-a2 per section (float32 values as printed)
    [4.708031187e-10, 9.416062374e-10, 4.708031187e-10, 9.084619880e-01, -0.0],
    [1.0, 2.0, 1.0, 1.832777858e+00, -8.412302136e-01],
    [1.0, 2.0, 1.0, 1.878710151e+00, -8.873743415e-01],
    [1.0, 1.0, 0.0, 1.949304223e+00, -9.582939744e-01]])


def _fw_sos():
    return np.hstack([ENV_LP_FIRMWARE_CMSIS[:, :3], np.ones((4, 1)), -ENV_LP_FIRMWARE_CMSIS[:, 3:5]])


class FirmwareToneSource(SampleSource):
    """1x sine (30 Hz, removed by the band-pass) + 5 kHz carrier AM-modulated at the NEW-bearing BPFO (4.9699 x 30 Hz)
    + small deterministic wide-band content; keyphasor exactly 1800 rpm. Units: g (100 mV/g, +-10 V ADC)."""
    F_BPFO = 4.9699 * 30.0

    def __init__(self, n_blocks=260):
        self.cfg, self.n_blocks = CoeConfig(), n_blocks

    def signal(self, t):
        rng = np.random.default_rng(11)
        am = (1 + 0.8 * np.cos(2 * np.pi * self.F_BPFO * t)) * 0.5 * np.sin(2 * np.pi * 5000 * t)
        return am + 0.2 * np.sin(2 * np.pi * 30 * t) + 0.01 * rng.standard_normal(t.size)

    def blocks(self):
        fs, nb, T = 30000.0, 256, 1.0 / 30.0
        x = self.signal(np.arange(self.n_blocks * nb) / fs)
        for s in range(self.n_blocks):
            codes = np.zeros((nb, 8), dtype=np.int16)
            codes[:, 0:4] = np.rint(x[s * nb:(s + 1) * nb, None] * 0.1 * 3276.8)
            k0, k1 = int(np.floor(s * nb / fs / T)) + 1, int(np.floor((s + 1) * nb / fs / T))
            yield SampleBlock(s, codes, np.array([k * T for k in range(k0, k1 + 1)]))


def firmware_reference(x_g, n0, n1, shaft_hz, orders, half_order):
    """Independent re-implementation of the node's per-window maths for ONE channel (g in, 8 features out)."""
    fs, dec, nfft = 30000.0, 6, 4096
    bp = signal.sosfilt(signal.butter(4, [2000, 8000], btype="bandpass", output="sos", fs=fs), x_g)
    env_full = signal.sosfilt(_fw_sos(), np.abs(bp))
    seg = bp[n0:n1]
    cnt = seg.size
    rms = np.sqrt(np.sum(seg ** 2) / cnt)
    d = seg - seg.mean()                                  # time_features.c: central moments (Pearson kurtosis)
    kurt = cnt * np.sum(d ** 4) / np.sum(d ** 2) ** 2
    crest = np.max(np.abs(seg)) / rms
    env = env_full[n0:n1][(np.arange(n0, n1) % dec) == 0]   # keep 1 in 6, phase continuous from stream start
    env_rms = np.sqrt(np.mean(env ** 2))
    m = env.size
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(m) / m)    # PERIODIC Hann
    X = np.fft.rfft(np.concatenate([(env - env.mean()) * w, np.zeros(nfft - m)]))
    fs_env, df = fs / dec, fs / dec / nfft
    psd = np.empty(nfft // 2 + 1)
    for k in range(psd.size):                             # unpack_spectrum: one-sided, Hann-power corrected
        p = abs(X[k]) ** 2
        psd[k] = (1.0 if k in (0, nfft // 2) else 2.0) * p / (fs_env * np.sum(w * w))
    bands = []
    for o in orders:                                      # frequency_features.c::calculate_band_rms
        c = o * shaft_hz
        hw = max(half_order * shaft_hz, df)
        first, last = int(np.ceil((c - hw) / df)), int(np.floor((c + hw) / df))
        bands.append(np.sqrt(sum(psd[b] for b in range(first, last + 1)) * df))
    return np.array([rms, kurt, crest, env_rms, *bands])


def test_envelope_lp_design_reproduces_firmware_coefficients():
    p = CoePipeline(sim())
    assert np.allclose(p._sos_lp, _fw_sos(), rtol=1e-6, atol=1e-9)


def test_window_features_match_firmware_formula_reference():
    src = FirmwareToneSource()
    w = run(src, 1)[0]
    assert w.valid and w.rpm == 1800.0
    fs = 30000.0
    x = src.signal(np.arange(src.n_blocks * 256) / fs)
    x = np.rint(x * 0.1 * 3276.8) / (0.1 * 3276.8)        # same ADC quantisation as the codes
    n0, n1 = int(np.ceil(w.t0_s * fs - 1e-9)), int(np.ceil(w.t20_s * fs - 1e-9))
    orders = [src.cfg.orders[f] for f in ("FTF", "BSF", "BPFO", "BPFI")]
    ref = firmware_reference(x, n0, n1, 30.0, orders, 0.08)
    got = w.features.reshape(4, 8)[0].astype(np.float64)
    assert np.allclose(got, ref, rtol=2e-4, atol=1e-7), (got, ref)
    assert got[6] > 20 * max(got[4], got[5], got[7])           # the AM tone sits in the BPFO band (new bearing)
    assert np.allclose(w.features.reshape(4, 8)[1], w.features.reshape(4, 8)[0], rtol=1e-6)   # 4 identical channels


def test_sim_b2x_is_copy_of_b1y():
    f = run(sim("outer_race", 0.7), 1)[0].features.reshape(4, 8)
    assert np.array_equal(f[2], f[1])
    assert not np.array_equal(f[3], f[1])
