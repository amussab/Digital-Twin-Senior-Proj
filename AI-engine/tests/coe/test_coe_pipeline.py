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


def sim(fault="healthy", sev=0.0, rpm=1750.0, seed=3, secs=3.0, **kw):
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
    _, h = signal.sosfreqz(p._sos_lp, worN=[100.0, 5000.0], fs=30000.0)
    g = 20 * np.log10(np.abs(h))
    assert abs(g[0]) < 0.1 and g[1] < -15.0


# ---- envelope recovers a known modulation frequency -----------------------------------------------
class AmSource(SampleSource):
    """5 kHz carrier, 100 Hz sinusoidal AM, Keyphasor exactly 1750 rpm."""

    def __init__(self, n_blocks=300):
        self.cfg, self.n_blocks = CoeConfig(), n_blocks
        self.cfg.orders = dict(self.cfg.orders, BPFO=100.0 / (1750 / 60))  # put the AM line in the BPFO band

    def blocks(self):
        fs, nb = 30000.0, 256
        T = 60.0 / 1750.0
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
@pytest.mark.parametrize("rpm", [1750.0, 3600.0])
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


def test_calibration_code_to_ms2():
    p = CoePipeline(sim())
    codes = np.zeros((256, 8), dtype=np.int16)
    codes[:, 0] = 3277                      # ~1.0 V on the +-10 V range = 10 g at 100 mV/g
    acc, _ = p.calibrate(codes)
    assert acc[0, 0] == pytest.approx(10 * G, rel=1e-3)
    acc5, _ = CoePipeline(sim(), CoeConfig(adc_range_v=5.0)).calibrate(codes)
    assert acc5[0, 0] == pytest.approx(5 * G, rel=1e-3)   # same code = half the voltage on +-5 V


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
    for rpm in (1750.0, 3600.0):
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
    ramp = lambda t: 1750.0 + 400.0 * t        # noqa: E731  accelerating
    pipe = CoePipeline(SimulatedRigSource("healthy", 0, 1750.0, seed=1, n_blocks=300, rpm_schedule=ramp),
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
    assert payload.decode(got[0]).rpm == pytest.approx(1750, rel=0.01)
    tx.close(); rx.close()
