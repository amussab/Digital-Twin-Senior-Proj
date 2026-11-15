"""Receive live feature windows from the COE node and run the AI model on each one.

Run:
    python inference_receiver.py [--port 8000]
Put the model in model_hook.py (setup() and predict()). Nothing else needs editing.

How it works:
- Every 2 s it announces this laptop on the network (UDP broadcast "COE-HOST <port>"
  to port 8001). The node's ESP32 hears it and sends every window here.
- Each window arrives as an HTTP POST /features (JSON). host_receiver.py checks it
  (same rules as the training recorder) and the reply goes back at once.
- A separate worker thread calls model_hook.predict() for each window, so a slow
  model never delays the node. If the model falls behind, the oldest waiting
  windows are dropped and counted.
Do not run the training recorder at the same time (same port, same node).
Standard library only (your model may need its own packages).
"""
import argparse
from collections import OrderedDict
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import queue
import socket
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

HERE = Path(__file__).resolve().parent


def load_module(name):
    """Load a file from this folder by path (python -I leaves it off sys.path)."""
    spec = importlib.util.spec_from_file_location(name, HERE / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


host_receiver = load_module('host_receiver')
model_hook = load_module('model_hook')

BEACON_PORT = 8001
BEACON_PERIOD_S = 2.0
QUEUE_LIMIT = 50  # windows waiting for the model (about 35 s at 250 RPM)


class WindowSink:
    """Stands in for host_receiver's output file: each new valid window lands here."""

    def __init__(self, windows, stats):
        self.windows, self.stats = windows, stats

    def open(self, mode='a', encoding=None):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def write(self, text):
        message = json.loads(text)
        window = {
            'epoch_received': time.time(),
            'device_id': message['device_id'], 'session_id': message['session_id'],
            'sequence': message['sequence'], 'replay': bool(message.get('replay', False)),
            'received_at': datetime.now().astimezone().isoformat(timespec='milliseconds'),
            'rpm': message['rpm'],
            'window_id': message['window_id'],
            'valid_channels': message['valid_channels'],
            'features': list(message['features']),
            'named': dict(zip(model_hook.FEATURE_NAMES, message['features'])),
        }
        self.stats['last_seen'] = time.monotonic()
        while True:
            try:
                self.windows.put_nowait(window)
                return
            except queue.Full:
                try:
                    self.windows.get_nowait()  # model too slow: drop the oldest
                    self.stats['dropped'] += 1
                except queue.Empty:
                    pass


class InferenceHandler(host_receiver.FeatureHandler):
    def do_POST(self):
        with self.server.feature_lock:  # one node packet at a time, as in host_receiver
            super().do_POST()

    def do_GET(self):
        body = b'COE inference receiver is running. Results are printed in its terminal.\n'
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # one line per window is printed by the worker instead


class InferenceServer(ThreadingHTTPServer):
    allow_reuse_address = os.name != 'nt'  # on Windows a second server must fail, not share the port
    daemon_threads = True


BACKEND_URL = os.environ.get('BACKEND_URL', 'http://127.0.0.1:5080').rstrip('/')
LOG_DIR = Path(os.environ.get('RIG_LOG_DIR', HERE.parents[1] / 'logs'))
FWD_QUEUE_LIMIT = 20
SIM_TAG = '[SIMULATION; rig-anchored replay]'


def backend_body(window, out, took_ms, contract):
    """Camel-case JSON for POST /api/live/prediction."""
    bearings = []
    for b in (out or {}).get('bearings', []):
        ok = b.get('status') == 'ok'
        bearings.append({'bearing': b['bearing'], 'status': b.get('status', 'warming_up'),
                         'faultClass': b.get('fault_class', 'warming_up'),
                         'classProbs': b.get('class_probs') if ok else None,
                         'confidence': b.get('confidence') if ok else None,
                         'healthIndex': b.get('health_index') if ok else None,
                         'healthStage': b.get('health_stage') if ok else None,
                         'onsetDetected': b.get('onset_detected') if ok else None,
                         'reason': b.get('reason'),
                         # S9 fields (NodeEngine contracts with a RUL table cadence only; absent -> null -> dashboard n/a)
                         'aiRulHours': b.get('ai_rul_hours'), 'physicsRulHours': b.get('physics_rul_hours'),
                         'minerD': b.get('miner_d'), 'physicsHealthIndex': b.get('physics_health_index'),
                         'physicsSource': b.get('physics_source'), 'rulValid': b.get('rul_valid'),
                         'rulNote': b.get('rul_note'), 'timeBase': b.get('time_base')})
    now = time.time()
    return {'seq': window['sequence'], 'windowId': window['window_id'], 'rpm': float(window['rpm']),
            'validChannels': window['valid_channels'], 'deviceId': window['device_id'],
            'sessionId': window['session_id'], 'contract': contract,
            # a table time base (NODE_TIME_BASE=table) exists only for the simulated rig_anchor replay: never label it as rig data
            'provenance': SIM_TAG if os.environ.get('NODE_TIME_BASE') == 'table' else
                          ('[REPLAY of recorded rig data]' if window['replay'] else '[LIVE rig data]'),
            # hybrid_rig provenance (non-visual): None for every other contract
            'rulSource': (out or {}).get('rul_source'), 'rulContract': (out or {}).get('rul_contract'),
            'classContract': (out or {}).get('class_contract'), 'rulTimescale': (out or {}).get('rul_timescale'),
            'features': window['features'], 'baselinePhase': (out or {}).get('baseline_phase'),
            'inferenceMs': round(took_ms, 3),
            'receivedAtMs': int(window['epoch_received'] * 1000), 'predictedAtMs': int(now * 1000),
            'postedAtMs': int(now * 1000), 'bearings': bearings}


def run_forwarder(fwd, stats, stop):
    """Posts results to the backend. Never blocks the model or the node ACK; failures are counted and dropped."""
    while not stop.is_set():
        try:
            body = fwd.get(timeout=0.5)
        except queue.Empty:
            continue
        body['postsDropped'] = stats['fwd_dropped']
        body['postedAtMs'] = int(time.time() * 1000)
        t = time.perf_counter()
        try:
            req = urllib.request.Request(BACKEND_URL + '/api/live/prediction', json.dumps(body).encode(),
                                         {'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=2) as resp:
                resp.read()
            stats['fwd_ok'] += 1
            stats['last_post_ms'] = (time.perf_counter() - t) * 1000
        except Exception as error:
            stats['fwd_dropped'] += 1
            if stats['fwd_dropped'] in (1, 10) or stats['fwd_dropped'] % 100 == 0:
                print(f"Backend {BACKEND_URL} unreachable ({type(error).__name__}); dropped {stats['fwd_dropped']} posts so far.", flush=True)


def run_model(windows, stats, stop, fwd):
    model = model_hook.setup()
    contract = os.environ.get('NODE_CONTRACT', 'v3e_node')
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"live_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    stats['log_path'] = str(log_path)
    if hasattr(model, 'meta'):
        print(f"HYBRID RIG: {json.dumps(model.meta())}", flush=True)
    print(f'Model ready. Logging every window + prediction to {log_path}. Backend: {BACKEND_URL}', flush=True)
    with open(log_path, 'a', encoding='utf-8') as log:
        while not stop.is_set():
            try:
                window = windows.get(timeout=0.5)
            except queue.Empty:
                continue
            started = time.perf_counter()
            out = None
            try:
                out, result = model_hook.predict_full(window, model)
            except Exception as error:  # keep receiving even if the model fails once
                result = f'MODEL ERROR: {type(error).__name__}: {error}'
            stats['done'] += 1
            took_ms = (time.perf_counter() - started) * 1000
            body = backend_body(window, out, took_ms, contract)
            e2e_pred_ms = (time.time() - window['epoch_received']) * 1000
            try:
                fwd.put_nowait(body)
            except queue.Full:
                stats['fwd_dropped'] += 1
            log.write(json.dumps({'received_at': window['received_at'], 'device_id': window['device_id'],
                                  'session_id': window['session_id'], 'sequence': window['sequence'],
                                  'window_id': window['window_id'], 'rpm': window['rpm'],
                                  'replay': window['replay'], 'contract': contract,
                                  'rul_source': (out or {}).get('rul_source'), 'rul_contract': (out or {}).get('rul_contract'),
                                  'rul_timescale': (out or {}).get('rul_timescale'), 'features': window['features'],
                                  'prediction': out, 'infer_ms': round(took_ms, 3),
                                  'recv_to_prediction_ms': round(e2e_pred_ms, 3),
                                  'last_post_ms': round(stats.get('last_post_ms', 0.0), 3),
                                  'fwd_dropped': stats['fwd_dropped']}) + '\n')
            log.flush()
            print(f"{window['received_at'][11:23]}  window {window['window_id']:>6}  "
                  f"{window['rpm']} RPM  ->  {result}   (recv->pred {e2e_pred_ms:.0f} ms)", flush=True)


def local_ipv4s():
    """IPv4 addresses of this laptop, without 127.x."""
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(('8.8.8.8', 80))  # sends nothing; only picks the route
            ips.add(probe.getsockname()[0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith('127.') and ip != '0.0.0.0')


def run_beacon(http_port, stop):
    """Every 2 s: 'COE-HOST <port>' to UDP 8001, broadcast on every network."""
    payload = f'COE-HOST {http_port}'.encode('ascii')
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        while not stop.is_set():
            targets = ['255.255.255.255'] + sorted({ip.rsplit('.', 1)[0] + '.255' for ip in local_ipv4s()})
            for target in targets:
                try:
                    sock.sendto(payload, (target, BEACON_PORT))
                except OSError:
                    pass  # a network may be down
            stop.wait(BEACON_PERIOD_S)


def watch_feed(stats, stop):
    """Say clearly when no data is arriving."""
    warned = False
    while not stop.wait(5.0):
        seen = stats['last_seen']
        idle = time.monotonic() - (seen or stats['started'])
        if idle > 5 and not warned:
            print('No data from the node: is this laptop on the Wi-Fi COE-NODE, and is the '
                  'node powered (ESP LED green)?' if seen is None else
                  f'No data for {idle:.0f} s: check the Wi-Fi COE-NODE and the node.', flush=True)
            warned = True
        elif idle <= 5:
            warned = False
        if stats['dropped'] != stats['dropped_reported']:
            print(f"Model too slow: {stats['dropped']} windows dropped so far.", flush=True)
            stats['dropped_reported'] = stats['dropped']


def main():
    parser = argparse.ArgumentParser(description='Live COE windows -> model_hook.predict().')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    windows = queue.Queue(maxsize=QUEUE_LIMIT)
    stats = {'last_seen': None, 'started': time.monotonic(), 'dropped': 0,
             'dropped_reported': 0, 'done': 0, 'fwd_ok': 0, 'fwd_dropped': 0}
    try:
        server = InferenceServer(('0.0.0.0', args.port), InferenceHandler)
    except OSError as error:
        raise SystemExit(f'Cannot use port {args.port} ({error}). '
                         'Is the training recorder or another receiver still running?')
    server.output = WindowSink(windows, stats)
    server.quiet = True  # the model worker prints one line per window instead
    server.seen = OrderedDict()
    server.feature_lock = threading.Lock()

    stop = threading.Event()
    fwd = queue.Queue(maxsize=FWD_QUEUE_LIMIT)
    threading.Thread(target=run_model, args=(windows, stats, stop, fwd), daemon=True).start()
    threading.Thread(target=run_forwarder, args=(fwd, stats, stop), daemon=True).start()
    threading.Thread(target=run_beacon, args=(server.server_port, stop), daemon=True).start()
    threading.Thread(target=watch_feed, args=(stats, stop), daemon=True).start()
    print(f'COE inference receiver on port {server.server_port}. Connect this laptop to the '
          f'Wi-Fi COE-NODE. Ctrl+C stops.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
        print(f"Stopped. Windows run through the model: {stats['done']}, "
              f"dropped: {stats['dropped']}. Backend posts ok: {stats['fwd_ok']}, dropped: {stats['fwd_dropped']}.", flush=True)


if __name__ == '__main__':
    main()
