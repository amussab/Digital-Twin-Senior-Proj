"""Rehearsal without hardware: POST recorded windows to the inference receiver exactly like the COE node.

    python replay_node.py <windows.jsonl | recorder.csv> [--host localhost] [--port 8000]
                          [--rpm-override 1800] [--speed 1] [--limit N] [--start N]

Cadence is 1200/rpm s per window (0.667 s at 1800 rpm) divided by --speed. Each request carries schema_version 1,
a fresh monotonically increasing sequence, a 16-hex session_id, device_id, and the Idempotency-Key header
(device-session-sequence). Missing fields are synthesised (listed on start). Messages carry "replay": true so the
dashboard labels them [REPLAY of recorded rig data] instead of [LIVE rig data]. Standard library only.
"""
import argparse
import csv
import json
import secrets
import sys
import time
import urllib.error
import urllib.request

CHANNELS = ('B1X', 'B1Y', 'B2X', 'B2Y')
FEATURES = ('rms', 'kurtosis', 'crest', 'envelope_rms', 'ftf', 'bsf', 'bpfo', 'bpfi')
NAMES = [f'{c}_{f}' for c in CHANNELS for f in FEATURES]


def load(path):
    rows = []
    if path.lower().endswith('.csv'):
        with open(path, newline='', encoding='utf-8') as handle:
            for r in csv.DictReader(handle):
                rows.append({'rpm': int(float(r['rpm'])) if r.get('rpm') else None,
                             'window_id': int(r['window_id']) if r.get('window_id') else None,
                             'valid_channels': int(r['valid_channels']) if r.get('valid_channels') else 15,
                             'device_id': r.get('device_id') or None,
                             'features': [float(r[n]) for n in NAMES]})
    else:
        with open(path, encoding='utf-8') as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source')
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--rpm-override', type=int)
    ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--no-replay-flag', action='store_true', help='omit the "replay" marker (then the dashboard says LIVE)')
    args = ap.parse_args()

    rows = load(args.source)[args.start:]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        sys.exit('no windows in source')
    synth = sorted({k for k in ('rpm', 'window_id', 'device_id', 'valid_channels', 'bearing_profile') for r in rows[:1]
                    if r.get(k) is None})
    print(f'{len(rows)} windows from {args.source}; synthesised fields: {synth or "none"}; '
          f'rpm override: {args.rpm_override}; speed x{args.speed}', flush=True)
    session = secrets.token_hex(8)
    url = f'http://{args.host}:{args.port}/features'
    ok = bad = dup = 0
    rtts = []
    t_next = time.perf_counter()
    for i, r in enumerate(rows):
        rpm = args.rpm_override or r.get('rpm') or 1800
        seq = i + 1
        message = {'schema_version': 1, 'sequence': seq, 'window_id': seq, 'rpm': int(rpm),
                   'bearing_profile': r.get('bearing_profile') or 1, 'valid_channels': r.get('valid_channels', 15),
                   'acceleration_unit': 'g', 'device_id': r.get('device_id') or 'coe-node-01',
                   'session_id': session, 'features': r['features']}
        if not args.no_replay_flag:
            message['replay'] = True
        body = json.dumps(message).encode()
        req = urllib.request.Request(url, body, {'Content-Type': 'application/json',
                                                 'Idempotency-Key': f"{message['device_id']}-{session}-{seq}"})
        t = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                reply = json.loads(resp.read())
            rtts.append((time.perf_counter() - t) * 1000)
            ok += 1
            dup += bool(reply.get('duplicate'))
        except (urllib.error.URLError, OSError) as error:
            bad += 1
            print(f'window {seq}: {error}', flush=True)
        if seq % 50 == 0:
            print(f'sent {seq}/{len(rows)}  ack p50 {sorted(rtts)[len(rtts) // 2]:.1f} ms' if rtts else f'sent {seq}', flush=True)
        t_next += (1200.0 / rpm) / args.speed
        delay = t_next - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
    if rtts:
        rtts.sort()
        print(f'done: ok {ok}, failed {bad}, duplicate {dup}; node-ACK RTT p50 {rtts[len(rtts) // 2]:.1f} ms, '
              f'p95 {rtts[int(len(rtts) * .95) - 1]:.1f} ms', flush=True)


if __name__ == '__main__':
    main()
