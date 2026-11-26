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
QUEUE_LIMIT = 50  # windows waiting for the model (about 34 s at 1750 RPM, 17 s at 3600 RPM)
LATENCY_LIMIT_MS = 500.0  # spec: end-to-end latency (window ready on the STM32 -> AI result)


def node_latency(message):
    """Schema 3: the node's latency parts in ms (STM32, UART, ESP32, Wi-Fi, sum); else None."""
    if message.get('schema_version') != 3:
        return None
    names = ('stm', 'uart', 'esp', 'wifi', 'node')
    return {f'{name}_ms': round(message[f'{name}_latency_us'] / 1000, 3) for name in names}


def digital_twin_view(window, received_s):
    """JSON for GET /digital-twin: one reading per window (speed, both probes), see the README."""
    flags = window['shaft_flags'] or 0
    view = {'available': True, 'age_s': round(time.monotonic() - received_s, 2),
            'received_at': window['received_at'], 'window_id': window['window_id'],
            'rpm': window['rpm'], 'measured_rpm': window['measured_rpm'],
            'revolutions': window['revolutions'], 'shaft_flags': window['shaft_flags'],
            'sample_rate_hz': window['sample_rate_hz'],
            # slowest / fastest turn in the window (schema 3; None before)
            'rpm_min': window['health'].get('rpm_min'), 'rpm_max': window['health'].get('rpm_max'),
            # node health (schema 3; None before): counters, Wi-Fi link and signal
            'node_health': ({name: window['health'][name] for name in
                             host_receiver.HEALTH_COUNTERS + ['wifi_link', 'wifi_rssi_dbm']}
                            if window['health'] else None)}
    for index, probe in enumerate(host_receiver.PROBES):
        values = {name: window['shaft'].get(f'{probe}_{name}') for name in host_receiver.PROBE_VALUES}
        view[probe] = dict(values, valid=bool(flags & (2 << index)), x1_valid=bool(flags & (8 << index)))
    view['node_latency'] = window['latency']  # None from older firmware
    return view


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
            # New firmware (schema 2): the node measures the speed itself (Keyphasor); 'rpm' above is that
            # measured speed in whole RPM. Old firmware (schema 1) has no measured_rpm / shaft values.
            'schema_version': message.get('schema_version'), 'bearing_profile': message.get('bearing_profile'),
            'measured_rpm': message.get('measured_rpm') if message.get('shaft_flags', 0) & 1 else None,
            'shaft_flags': message.get('shaft_flags'),
            'revolutions': message.get('revolutions'),
            'shaft': {name: message[name] for name in host_receiver.SHAFT_FIELDS if name in message},
            # ADC samples per second per channel (schema 3 firmware; None before), e.g. 29998.5
            'sample_rate_hz': message.get('sample_rate_hz'),
            # Speed range and node health (schema 3; {} before), see host_receiver.NODE_FIELDS
            'health': {name: message[name] for name in host_receiver.NODE_FIELDS if name in message},
            # End-to-end latency chain (schema 3 firmware; None before). host_started is when
            # this laptop got the POST; the worker adds its own time to the node's part.
            'latency': node_latency(message),
            'host_started': self.stats.get('post_started') or time.perf_counter(),
        }
        self.stats['last_seen'] = time.monotonic()
        self.stats['latest'] = (window, self.stats['last_seen'])  # for GET /digital-twin, before the model runs
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
        started = time.perf_counter()  # start of this laptop's part of the end-to-end latency
        with self.server.feature_lock:  # one node packet at a time, as in host_receiver
            self.server.output.stats['post_started'] = started
            super().do_POST()

    def do_GET(self):
        if self.path.split('?')[0] == '/digital-twin':
            stats = self.server.output.stats
            latest = stats.get('latest')
            data = digital_twin_view(*latest) if latest else {'available': False}
            data['ai_result'] = stats.get('ai_result')
            body = json.dumps(data).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Access-Control-Allow-Origin', '*')  # a web page on another port may read it
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = (b'COE inference receiver is running. Results are printed in its terminal.\n'
                b'Digital twin data (speed, proximity probes): GET /digital-twin\n')
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


def probes_body(window):
    """1X probe readings for the ME FE twin in the backend (POST /api/live/prediction 'probes').
    None from old firmware (schema 1). Units as sent by the node: um pk-pk, phase lag in degrees."""
    flags = window.get('shaft_flags')
    if flags is None:
        return None
    shaft = window['shaft']
    body = {'measuredRpm': window['measured_rpm']}
    for index, (probe, key) in enumerate(zip(host_receiver.PROBES, ('p1', 'p3'))):
        valid, x1 = bool(flags & (2 << index)), bool(flags & (8 << index))
        body[f'{key}GapUm'] = shaft.get(f'{probe}_gap_um') if valid else None
        body[f'{key}X1UmPp'] = shaft.get(f'{probe}_x1_um_pp') if x1 else None
        body[f'{key}X1PhaseDeg'] = shaft.get(f'{probe}_x1_phase_deg') if x1 else None
        body[f'{key}X1Valid'] = x1
    return body


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
            'postedAtMs': int(now * 1000), 'bearings': bearings,
            'probes': probes_body(window)}


def shaft_text(window):
    """Short text of the Keyphasor speed and probe gaps for the terminal line."""
    if window['shaft_flags'] is None:
        return f"{window['rpm']} RPM (old firmware: no Keyphasor or probe values)"
    speed = (f"{window['measured_rpm']:.1f} RPM ({window['revolutions']} turns)"
             if window['measured_rpm'] is not None else f"{window['rpm']} RPM (no Keyphasor speed)")
    gaps = []
    for index, probe in enumerate(host_receiver.PROBES):
        valid = window['shaft_flags'] & (2 << index)
        gaps.append(f"{probe.upper()} gap {window['shaft'][probe + '_gap_um']:.0f} um" if valid
                    else f'{probe.upper()} not valid')
    return speed + ', ' + ', '.join(gaps)


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
            finished = time.perf_counter()
            stats['done'] += 1
            took_ms = (finished - started) * 1000
            host_ms = (finished - window['host_started']) * 1000  # POST received -> AI result
            node = window['latency']
            end_to_end_ms = round(node['node_ms'] + host_ms, 1) if node else None
            stats['ai_result'] = {'window_id': window['window_id'], 'result': str(result),
                                  'host_latency_ms': round(host_ms, 1), 'end_to_end_ms': end_to_end_ms,
                                  'latency_limit_ms': LATENCY_LIMIT_MS,
                                  'within_limit': None if end_to_end_ms is None
                                  else end_to_end_ms < LATENCY_LIMIT_MS}
            if end_to_end_ms is None:
                latency = 'end-to-end: needs schema 3 firmware'
            else:
                stats['latency_count'] = stats.get('latency_count', 0) + 1
                stats['latency_max_ms'] = max(stats.get('latency_max_ms', 0.0), end_to_end_ms)
                stats['latency_over'] = stats.get('latency_over', 0) + (end_to_end_ms >= LATENCY_LIMIT_MS)
                latency = (f"end-to-end {end_to_end_ms:.0f} ms "
                           f"{'OK' if end_to_end_ms < LATENCY_LIMIT_MS else 'OVER'} (limit 500: "
                           f"STM32 {node['stm_ms']:.0f} + UART {node['uart_ms']:.0f} + ESP32 {node['esp_ms']:.0f}"
                           f" + Wi-Fi {node['wifi_ms']:.0f} + laptop {host_ms:.0f})")
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
                                  'fwd_dropped': stats['fwd_dropped'],
                                  'node_latency': node, 'end_to_end_ms': end_to_end_ms}) + '\n')
            log.flush()
            print(f"{window['received_at'][11:23]}  window {window['window_id']:>6}  "
                  f"{shaft_text(window)}  ->  {result}   (recv->pred {e2e_pred_ms:.0f} ms; {latency})", flush=True)
            health_warning(window['health'], stats)


def health_warning(health, stats):
    """Print a line when a node health counter goes up (counts restart when the node restarts)."""
    if not health:
        return
    last = stats.get('health_last') or {}
    rising = [f"{name.replace('_', ' ')} {health[name]}" for name in host_receiver.HEALTH_COUNTERS
              if health[name] > last.get(name, 0)]
    stats['health_last'] = health
    if rising:
        print(f"NODE HEALTH: {', '.join(rising)} (counts since the node started; Wi-Fi "
              f"{health['wifi_link']}, signal {health['wifi_rssi_dbm']} dBm)", flush=True)


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
             'dropped_reported': 0, 'done': 0, 'fwd_ok': 0, 'fwd_dropped': 0, 'latest': None, 'ai_result': None}
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
        if stats.get('latency_count'):
            print(f"End-to-end latency: max {stats['latency_max_ms']:.0f} ms over "
                  f"{stats['latency_count']} windows, {stats['latency_over']} at or over "
                  f"{LATENCY_LIMIT_MS:.0f} ms.", flush=True)


if __name__ == '__main__':
    main()
