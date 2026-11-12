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
        stats['done'] += 1
        took_ms = (time.perf_counter() - started) * 1000
        print(f"{window['received_at'][11:23]}  window {window['window_id']:>6}  "
              f"{window['rpm']} RPM  ->  {result}   ({took_ms:.0f} ms)", flush=True)


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
             'dropped_reported': 0, 'done': 0}
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
              f"dropped: {stats['dropped']}.", flush=True)


if __name__ == '__main__':
    main()
