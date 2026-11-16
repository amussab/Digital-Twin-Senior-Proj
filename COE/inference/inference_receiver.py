"""Receive live feature windows from the COE node and run the AI model on each one.

Run:
    python inference_receiver.py [--port 8000]
The machine speed is not set here: the node measures it from the Keyphasor (ADC
channel 7) and computes the features for that speed. Each window also carries the
measured speed and the two proximity probe values (for the digital twin).
Put the model in model_hook.py (setup() and predict()). Nothing else needs editing.

How it works:
- Every 2 s it announces this laptop on the network (UDP broadcast
  "COE-HOST <port>" to port 8001). The node's ESP32 hears it and sends every window here.
- Each window arrives as an HTTP POST /features (JSON). host_receiver.py checks it
  (same rules as the training recorder) and the reply goes back at once.
- A separate worker thread calls model_hook.predict() for each window, so a slow
  model never delays the node. If the model falls behind, the oldest waiting
  windows are dropped and counted.
- GET /digital-twin returns the newest window's machine speed and proximity probe values
  (and the model's latest answer) as JSON, for the digital twin's user interface.
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
            'received_at': datetime.now().astimezone().isoformat(timespec='milliseconds'),
            'rpm': message['rpm'],
            'window_id': message['window_id'],
            'valid_channels': message['valid_channels'],
            'features': list(message['features']),
            'named': dict(zip(model_hook.FEATURE_NAMES, message['features'])),
            # Keyphasor speed and probe values (schema 2 firmware). None / {} from old firmware.
            # A value whose shaft_flags bit is clear is 0 (see host_receiver.validate_shaft).
            'measured_rpm': message.get('measured_rpm') if message.get('shaft_flags', 0) & 1 else None,
            'revolutions': message.get('revolutions'),
            'shaft_flags': message.get('shaft_flags'),
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


def run_model(windows, stats, stop):
    model = model_hook.setup()
    print('Model ready (model_hook.setup() done). Waiting for windows...', flush=True)
    while not stop.is_set():
        try:
            window = windows.get(timeout=0.5)
        except queue.Empty:
            continue
        started = time.perf_counter()
        try:
            result = model_hook.predict(window, model)
        except Exception as error:  # keep receiving even if the model fails once
            result = f'MODEL ERROR: {type(error).__name__}: {error}'
        finished = time.perf_counter()
        took_ms = (finished - started) * 1000
        host_ms = (finished - window['host_started']) * 1000  # POST received -> AI result
        node = window['latency']
        end_to_end_ms = round(node['node_ms'] + host_ms, 1) if node else None
        stats['done'] += 1
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
        print(f"{window['received_at'][11:23]}  window {window['window_id']:>6}  "
              f"{shaft_text(window)}  ->  {result}   (model {took_ms:.0f} ms; {latency})", flush=True)
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
             'dropped_reported': 0, 'done': 0, 'latest': None, 'ai_result': None}
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
    threading.Thread(target=run_model, args=(windows, stats, stop), daemon=True).start()
    threading.Thread(target=run_beacon, args=(server.server_port, stop), daemon=True).start()
    threading.Thread(target=watch_feed, args=(stats, stop), daemon=True).start()
    print(f'COE inference receiver on port {server.server_port} (the node measures the machine '
          f'speed itself). Connect this laptop to the Wi-Fi COE-NODE. Ctrl+C stops.\n'
          f'Digital twin: latest speed and probe values at http://localhost:{server.server_port}/digital-twin',
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
        print(f"Stopped. Windows run through the model: {stats['done']}, "
              f"dropped: {stats['dropped']}.", flush=True)
        if stats.get('latency_count'):
            print(f"End-to-end latency: max {stats['latency_max_ms']:.0f} ms over "
                  f"{stats['latency_count']} windows, {stats['latency_over']} at or over "
                  f"{LATENCY_LIMIT_MS:.0f} ms.", flush=True)


if __name__ == '__main__':
    main()
