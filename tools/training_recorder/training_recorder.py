"""Record labelled training data from the COE bearing node, with a small web UI.

Run (from any folder):
    python -I training_recorder.py [--port 8000] [--data-dir PATH]
Then open http://localhost:8000/ in a browser on this laptop.

The ESP32 keeps posting to http://<laptop>:8000/features. That endpoint is the
unchanged FeatureHandler from tests/host_receiver.py (same validation, dedup,
status codes and replies). Every valid packet is acknowledged; it is saved to
the current run only while recording is on.

Files in the data folder, one pair per run:
    run_YYYYMMDD_HHMMSS.json   label, note, start and stop time
    run_YYYYMMDD_HHMMSS.jsonl  one line per window: {"received_at": ..., "message": {...}}
While running it also broadcasts a UDP beacon "COE-HOST <http_port>" to port 8001
every 2 s, so the ESP32 can find this laptop without a fixed IP.
Standard library only.
"""
import argparse
import csv
from datetime import datetime
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import socket
import threading
import time
from collections import OrderedDict
from http.server import ThreadingHTTPServer

HERE = Path(__file__).resolve().parent

# Load host_receiver.py by path (python -I does not put the script folder on sys.path):
# next to this script (team repo copy) or in ../tests (coe_esp32).
_receiver = HERE / 'host_receiver.py'
if not _receiver.exists():
    _receiver = HERE.parent / 'tests' / 'host_receiver.py'
_spec = importlib.util.spec_from_file_location('host_receiver', _receiver)
host_receiver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(host_receiver)

LABELS = [
    ('healthy', 'Healthy'),
    ('inner_race', 'Inner race fault'),
    ('outer_race', 'Outer race fault'),
    ('ball', 'Ball (rolling element) fault'),
    ('cage', 'Cage fault'),
]
LABEL_NAMES = dict(LABELS)

# Channel-major order, see coe_firmware/README.md section 6.
CHANNELS = ('B1X', 'B1Y', 'B2X', 'B2Y')
FEATURES = ('rms', 'kurtosis', 'crest', 'envelope_rms', 'ftf', 'bsf', 'bpfo', 'bpfi')
FEATURE_COLUMNS = [f'{channel}_{feature}' for channel in CHANNELS for feature in FEATURES]
CSV_COLUMNS = ['run_id', 'label', 'note', 'received_at', 'rpm', 'window_id', 'sequence',
               'device_id', 'valid_channels'] + FEATURE_COLUMNS

BEACON_PORT = 8001
BEACON_PERIOD_S = 2.0
RUN_ID = re.compile(r'run_\d{8}_\d{6}(_\d{1,3})?')


def now_iso():
    return datetime.now().astimezone().isoformat(timespec='milliseconds')


def epoch(iso):
    return datetime.fromisoformat(iso).timestamp()


class Recorder:
    """Owns the runs on disk and the recording state. All changes go through self.lock."""

    def __init__(self, data_dir):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.runs = {}            # run_id -> meta dict plus 'windows' and 'rpms'
        self.active = None        # run_id being recorded, or None
        self.last_packet = None   # (time.monotonic(), message) of the last new valid window
        self.beacon_ips = []      # laptop IPv4 addresses the beacon currently announces
        self._load()

    def _meta_path(self, run_id):
        return self.dir / f'{run_id}.json'

    def _rows_path(self, run_id):
        return self.dir / f'{run_id}.jsonl'

    def _write_meta(self, meta):
        data = {key: meta[key] for key in ('run_id', 'label', 'note', 'started_at', 'stopped_at')}
        path = self._meta_path(meta['run_id'])
        temp = path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(data, indent=2), encoding='utf-8')
        os.replace(temp, path)

    def rows(self, run_id):
        """Yield saved windows of a run; skip a line cut short by a crash."""
        path = self._rows_path(run_id)
        if not path.exists():
            return
        with path.open(encoding='utf-8') as source:
            for line in source:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and isinstance(row.get('message'), dict):
                    yield row

    def _load(self):
        for meta_path in sorted(self.dir.glob('run_*.json')):
            run_id = meta_path.stem
            if not RUN_ID.fullmatch(run_id):
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding='utf-8'))
                if meta.get('label') not in LABEL_NAMES:
                    raise ValueError('unknown label')
                meta['run_id'] = run_id
                meta.setdefault('note', '')
                windows, rpms, last = 0, set(), None
                for row in self.rows(run_id):
                    windows += 1
                    rpms.add(row['message'].get('rpm'))
                    last = row.get('received_at')
                if not meta.get('stopped_at'):
                    # The script was closed while this run was recording: close it now.
                    meta['stopped_at'] = last or meta['started_at']
                    self._write_meta(meta)
                meta['windows'], meta['rpms'] = windows, rpms
                self.runs[run_id] = meta
            except (OSError, ValueError, KeyError, TypeError) as error:
                print(f'Skipping {meta_path.name}: {error}', flush=True)

    def start(self, label, note):
        if label not in LABEL_NAMES:
            raise ValueError('Unknown label')
        note = ' '.join(str(note or '').split())[:200]
        with self.lock:
            if self.active:
                raise RuntimeError('Already recording. Stop first.')
            base = 'run_' + datetime.now().strftime('%Y%m%d_%H%M%S')
            run_id, number = base, 1
            while run_id in self.runs or self._meta_path(run_id).exists():
                number += 1
                run_id = f'{base}_{number}'
            meta = {'run_id': run_id, 'label': label, 'note': note,
                    'started_at': now_iso(), 'stopped_at': None}
            self._write_meta(meta)
            self._rows_path(run_id).touch()
            meta['windows'], meta['rpms'] = 0, set()
            self.runs[run_id] = meta
            self.active = run_id
            print(f'Recording started: {run_id} label={label}', flush=True)
            return run_id

    def stop(self):
        with self.lock:
            if not self.active:
                raise RuntimeError('Not recording')
            meta = self.runs[self.active]
            meta['stopped_at'] = now_iso()
            self._write_meta(meta)
            print(f"Recording stopped: {self.active}, windows={meta['windows']}", flush=True)
            self.active = None

    def delete(self, run_id):
        with self.lock:
            if run_id not in self.runs:
                raise KeyError('No such run')
            if run_id == self.active:
                raise RuntimeError('Stop recording before deleting this run')
            self._rows_path(run_id).unlink(missing_ok=True)
            self._meta_path(run_id).unlink(missing_ok=True)
            del self.runs[run_id]
            print(f'Deleted {run_id}', flush=True)

    def record(self, canonical_line):
        """Called once per new valid window. An OSError here makes the handler reply 500."""
        message = json.loads(canonical_line)
        with self.lock:
            self.last_packet = (time.monotonic(), message)
            if not self.active:
                return
            row = json.dumps({'received_at': now_iso(), 'message': message}, sort_keys=True)
            with self._rows_path(self.active).open('a', encoding='utf-8') as output:
                output.write(row + '\n')
            meta = self.runs[self.active]
            meta['windows'] += 1
            meta['rpms'].add(message['rpm'])

    def status(self):
        with self.lock:
            now = time.time()
            runs = []
            for meta in sorted(self.runs.values(), key=lambda m: m['started_at'], reverse=True):
                start = epoch(meta['started_at'])
                end = epoch(meta['stopped_at']) if meta['stopped_at'] else now
                runs.append({
                    'run_id': meta['run_id'], 'label': meta['label'],
                    'label_name': LABEL_NAMES[meta['label']], 'note': meta['note'],
                    'started': datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S'),
                    'duration_s': round(max(0.0, end - start), 1), 'windows': meta['windows'],
                    'rpm': ', '.join(str(r) for r in sorted(r for r in meta['rpms'] if r is not None)),
                    'recording': meta['run_id'] == self.active,
                })
            summary = {key: 0 for key, _ in LABELS}
            for meta in self.runs.values():
                summary[meta['label']] += meta['windows']
            last = None
            if self.last_packet:
                seen, message = self.last_packet
                last = {'age_s': round(time.monotonic() - seen, 1), 'rpm': message.get('rpm'),
                        'valid_channels': message.get('valid_channels'),
                        'device_id': message.get('device_id')}
            active = next((run for run in runs if run['recording']), None)
            return {'recording': active, 'last_packet': last, 'runs': runs, 'summary': summary,
                    'beacon_ips': list(self.beacon_ips)}

    def csv_bytes(self, run_ids):
        with self.lock:
            metas = [dict(self.runs[r]) for r in run_ids if r in self.runs]
        metas.sort(key=lambda m: m['started_at'])
        text = io.StringIO()
        writer = csv.writer(text, lineterminator='\n')
        writer.writerow(CSV_COLUMNS)
        for meta in metas:
            for row in self.rows(meta['run_id']):
                message = row['message']
                writer.writerow([meta['run_id'], meta['label'], meta['note'], row.get('received_at'),
                                 message.get('rpm'), message.get('window_id'), message.get('sequence'),
                                 message.get('device_id'), message.get('valid_channels')]
                                + list(message.get('features', [])))
        return text.getvalue().encode('utf-8')


class RunSink:
    """Stands in for host_receiver's output Path. FeatureHandler calls
    output.open('a', ...).write(canonical + '\\n') once per new valid vector."""

    def __init__(self, recorder):
        self.recorder = recorder

    def open(self, mode='a', encoding=None):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def write(self, text):
        self.recorder.record(text)


class RecorderHandler(host_receiver.FeatureHandler):
    """POST /features goes to the original handler; everything else is the UI."""

    timeout = 30  # close idle browser connections so they cannot hold a thread forever

    def do_POST(self):
        if self.path.startswith('/api/'):
            self.api_post()
            return
        with self.server.feature_lock:  # one ESP packet at a time, as in host_receiver
            super().do_POST()

    def api_post(self):
        recorder = self.server.recorder
        try:
            length = int(self.headers.get('Content-Length', '0'))
            # JSON content type also blocks cross-site form posts from other web pages.
            if self.headers.get_content_type() != 'application/json' or not 0 <= length <= 2048:
                raise ValueError('Expected a small JSON body')
            body = json.loads(self.rfile.read(length) or b'{}')
            if not isinstance(body, dict):
                raise ValueError('Expected a JSON object')
            if self.path == '/api/start':
                run_id = recorder.start(body.get('label'), body.get('note', ''))
                self.reply(200, {'ok': True, 'run_id': run_id})
            elif self.path == '/api/stop':
                recorder.stop()
                self.reply(200, {'ok': True})
            elif self.path == '/api/delete':
                recorder.delete(str(body.get('run_id', '')))
                self.reply(200, {'ok': True})
            else:
                self.reply(404, {'ok': False, 'error': 'Not found'})
        except RuntimeError as error:
            self.reply(409, {'ok': False, 'error': str(error)})
        except KeyError as error:
            self.reply(404, {'ok': False, 'error': str(error.args[0])})
        except (ValueError, TypeError) as error:
            self.reply(400, {'ok': False, 'error': str(error)})
        except OSError as error:
            self.reply(500, {'ok': False, 'error': f'Disk error: {error}'})

    def do_GET(self):
        recorder = self.server.recorder
        path = self.path.split('?', 1)[0]
        if path == '/':
            self.send_bytes(PAGE.encode('utf-8'), 'text/html; charset=utf-8')
        elif path == '/api/status':
            self.send_bytes(json.dumps(recorder.status()).encode('utf-8'), 'application/json')
        elif path == '/export/all.csv':
            with recorder.lock:
                run_ids = list(recorder.runs)
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.send_bytes(recorder.csv_bytes(run_ids), 'text/csv; charset=utf-8',
                            f'training_all_{stamp}.csv')
        elif (match := re.fullmatch(r'/export/(run_[0-9_]+)\.csv', path)) and match[1] in recorder.runs:
            self.send_bytes(recorder.csv_bytes([match[1]]), 'text/csv; charset=utf-8',
                            f'training_{match[1]}.csv')
        else:
            self.send_bytes(b'Not found', 'text/plain', status=404)

    def send_bytes(self, body, content_type, filename=None, status=200):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        if filename:
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        if getattr(self, 'command', None) != 'GET':  # the UI polls every second; keep the log readable
            super().log_message(format, *args)


class RecorderServer(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR would let a second server silently share port 8000.
    allow_reuse_address = os.name != 'nt'
    daemon_threads = True


def local_ipv4s():
    """IPv4 addresses of this machine (hostname lookup plus the default-route address), no 127.x."""
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(('8.8.8.8', 80))  # UDP connect sends nothing; it only picks the route
            ips.add(probe.getsockname()[0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith('127.') and ip != '0.0.0.0')


def run_beacon(http_port, recorder, stop):
    """Every 2 s send b'COE-HOST <port>' to 255.255.255.255 and each x.y.z.255, UDP port 8001."""
    payload = f'COE-HOST {http_port}'.encode('ascii')
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        while not stop.is_set():
            ips = local_ipv4s()
            recorder.beacon_ips = ips
            targets = ['255.255.255.255'] + sorted({ip.rsplit('.', 1)[0] + '.255' for ip in ips})
            for target in targets:
                try:
                    sock.sendto(payload, (target, BEACON_PORT))
                except OSError:
                    pass  # an interface may be down
            stop.wait(BEACON_PERIOD_S)


def make_server(port, recorder, bind='0.0.0.0'):
    server = RecorderServer((bind, port), RecorderHandler)
    server.recorder = recorder
    server.output = RunSink(recorder)   # same attributes host_receiver.make_server sets
    server.seen = OrderedDict()
    server.feature_lock = threading.Lock()
    return server


PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bearing training recorder</title>
<style>
:root { --bg:#f3f5f8; --card:#fff; --text:#1c2631; --muted:#5b6979; --line:#d9e0e8; --go:#1e8449; --stop:#c0392b; }
* { box-sizing:border-box; }
body { margin:0; font:16px/1.45 "Segoe UI", system-ui, sans-serif; background:var(--bg); color:var(--text); }
header { display:flex; justify-content:space-between; align-items:center; gap:16px; padding:14px 28px; background:#1c2631; color:#fff; }
header h1 { margin:0; font-size:20px; font-weight:600; }
.beacon { margin-left:auto; color:#c6d0db; font-size:14px; text-align:right; }
.feed { padding:6px 16px; border-radius:999px; font-weight:600; font-size:15px; background:#6b7785; white-space:nowrap; }
.feed.ok { background:var(--go); }
.feed.bad { background:var(--stop); }
main { max-width:1120px; margin:0 auto; padding:22px 28px 48px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:20px 22px; margin-bottom:18px; }
h2 { margin:0 0 14px; font-size:17px; font-weight:600; color:var(--muted); }
.modes { display:grid; grid-template-columns:repeat(5, 1fr); gap:12px; }
.mode { min-height:86px; padding:14px 10px; border:3px solid var(--line); border-radius:12px; background:#fff;
        color:var(--text); font:600 17px "Segoe UI", system-ui, sans-serif; cursor:pointer; }
.mode:hover:not(:disabled) { border-color:var(--c); }
.mode.sel { background:var(--c); border-color:var(--c); color:#fff; box-shadow:0 4px 14px rgba(0,0,0,.18); }
.mode:disabled { cursor:not-allowed; }
.mode:disabled:not(.sel) { opacity:.35; }
.selected { margin-top:14px; font-size:18px; }
.selected b { color:var(--c); }
.hint { color:var(--muted); font-size:14px; margin-left:8px; }
.rec-row { display:flex; gap:20px; align-items:flex-end; }
.note { flex:1; }
.note label { display:block; font-weight:600; color:var(--muted); margin-bottom:6px; }
.note input { width:100%; padding:12px 14px; font-size:16px; border:2px solid var(--line); border-radius:10px; }
.note input:disabled { background:#eef1f4; }
.toggle { min-width:280px; padding:20px 28px; border:0; border-radius:12px; color:#fff; font:700 22px "Segoe UI", system-ui, sans-serif; cursor:pointer; }
.toggle.start { background:var(--go); }
.toggle.stop { background:var(--stop); }
.toggle:disabled { opacity:.6; cursor:wait; }
.banner { margin-top:18px; padding:14px 18px; border-radius:10px; font-size:20px; font-weight:700; background:#e9edf1; color:var(--muted); }
.banner.on { background:var(--stop); color:#fff; }
.dot { display:inline-block; width:14px; height:14px; margin-right:10px; border-radius:50%; background:#fff; animation:blink 1s infinite; }
@keyframes blink { 50% { opacity:.15; } }
.stats { display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; margin-top:14px; }
.stats div { padding:12px 16px; border:1px solid var(--line); border-radius:10px; }
.stats span { display:block; font-size:14px; color:var(--muted); }
.stats b { font-size:26px; font-variant-numeric:tabular-nums; }
.stats b.bad { color:var(--stop); font-size:20px; }
.msg { margin-top:12px; color:var(--stop); font-weight:600; min-height:1.4em; }
.head { display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; }
.head h2 { margin:0; }
.btn { display:inline-block; padding:10px 18px; border-radius:10px; background:#1f5fa8; color:#fff; font-weight:600; text-decoration:none; }
.btn.off { background:#9aa6b2; pointer-events:none; }
.summary { display:grid; grid-template-columns:repeat(5, 1fr); gap:12px; }
.sum { padding:10px 12px; border:1px solid var(--line); border-radius:10px; font-size:14px; }
.sum b { display:block; font-size:22px; font-variant-numeric:tabular-nums; }
.bar { height:6px; margin-top:6px; border-radius:3px; background:#e6eaef; }
.bar i { display:block; height:100%; border-radius:3px; background:var(--c); }
table { width:100%; border-collapse:collapse; font-size:15px; }
th, td { padding:9px 10px; border-bottom:1px solid var(--line); text-align:left; }
th { font-size:13px; color:var(--muted); text-transform:uppercase; letter-spacing:.03em; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
.tag { display:inline-block; padding:2px 10px; border-radius:999px; background:var(--c); color:#fff; font-weight:600; font-size:14px; }
.live { color:var(--stop); font-weight:700; }
td a { color:#1f5fa8; font-weight:600; }
.del { padding:5px 12px; border:1px solid var(--stop); border-radius:8px; background:#fff; color:var(--stop); font-weight:600; cursor:pointer; }
.empty { color:var(--muted); padding:16px 4px; }
@media (max-width:900px) { .modes, .summary { grid-template-columns:repeat(2, 1fr); } .rec-row { flex-direction:column; align-items:stretch; } }
</style>
</head>
<body>
<header>
  <h1>Bearing training data recorder</h1>
  <div id="beacon" class="beacon"></div>
  <div id="feed" class="feed">Connecting...</div>
</header>
<main>
  <section class="card">
    <h2>1. Choose the bearing condition</h2>
    <div id="modes" class="modes"></div>
    <div id="selected" class="selected"></div>
  </section>

  <section class="card">
    <h2>2. Record</h2>
    <div class="rec-row">
      <div class="note">
        <label for="note">Note for this run (optional)</label>
        <input id="note" maxlength="120" placeholder="e.g. load 50 %, sensor moved">
      </div>
      <button id="toggle" class="toggle start">Start recording</button>
    </div>
    <div id="banner" class="banner">Not recording</div>
    <div class="stats">
      <div><span>Elapsed</span><b id="elapsed">-</b></div>
      <div><span>Windows saved in this run</span><b id="count">-</b></div>
      <div><span>Last packet from the node</span><b id="last">-</b></div>
    </div>
    <div id="msg" class="msg"></div>
  </section>

  <section class="card">
    <div class="head">
      <h2>3. Recorded runs</h2>
      <a id="all" class="btn" href="/export/all.csv">Download all runs (CSV)</a>
    </div>
    <div class="hint" style="margin:0 0 8px">Windows saved per label (check the classes are balanced)</div>
    <div id="summary" class="summary"></div>
    <div id="runs" style="margin-top:16px"></div>
  </section>
</main>
<script>
const LABELS = __LABELS__;
const COLORS = {healthy:'#1e8449', inner_race:'#d35400', outer_race:'#7d3c98', ball:'#1f5fa8', cage:'#117a8b'};
const NAME = Object.fromEntries(LABELS);
const $ = id => document.getElementById(id);
let selected = NAME[localStorage.getItem('label')] ? localStorage.getItem('label') : 'healthy';
let state = null, runsKey = '', busy = false;

function esc(text) { const d = document.createElement('div'); d.textContent = text == null ? '' : String(text); return d.innerHTML; }
function dur(s) {
  s = Math.max(0, Math.floor(s));
  const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), x = String(s % 60).padStart(2, '0');
  return h ? h + ':' + String(m).padStart(2, '0') + ':' + x : m + ':' + x;
}
async function post(path, body) {
  const r = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body || {})});
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.error || ('HTTP ' + r.status));
  return d;
}

for (const [key, name] of LABELS) {
  const b = document.createElement('button');
  b.className = 'mode'; b.textContent = name; b.dataset.key = key; b.style.setProperty('--c', COLORS[key]);
  b.onclick = () => { selected = key; localStorage.setItem('label', key); render(); };
  $('modes').appendChild(b);
}

$('toggle').onclick = async () => {
  if (!state || busy) return;
  busy = true; $('toggle').disabled = true; $('msg').textContent = '';
  try {
    if (state.recording) await post('/api/stop');
    else await post('/api/start', {label:selected, note:$('note').value});
  } catch (e) { $('msg').textContent = e.message; }
  busy = false; $('toggle').disabled = false;
  refresh();
};

async function del(id, text) {
  if (!confirm('Delete this run?\n\n' + text + '\n\nIts data will be removed from disk. This cannot be undone.')) return;
  try { await post('/api/delete', {run_id:id}); } catch (e) { $('msg').textContent = e.message; }
  refresh();
}

function render() {
  const rec = state && state.recording;
  const shown = rec ? rec.label : selected;
  for (const b of $('modes').children) {
    b.classList.toggle('sel', b.dataset.key === shown);
    b.disabled = !!rec;
  }
  $('selected').style.setProperty('--c', COLORS[shown]);
  $('selected').innerHTML = 'Selected: <b>' + esc(NAME[shown]) + '</b>' +
    (rec ? '<span class="hint">Stop recording to change the condition.</span>' : '');
  if (!state) return;

  $('beacon').textContent = state.beacon_ips.length
    ? 'Announcing this laptop as ' + state.beacon_ips.map(ip => ip + ':' + (location.port || '80')).join(', ')
    : 'Beacon: no network address found';
  const t = $('toggle');
  t.textContent = rec ? 'Stop recording' : 'Start recording';
  t.className = 'toggle ' + (rec ? 'stop' : 'start');
  $('note').disabled = !!rec;
  if (rec && document.activeElement !== $('note')) $('note').value = rec.note;
  $('banner').className = 'banner' + (rec ? ' on' : '');
  $('banner').innerHTML = rec ? '<span class="dot"></span>RECORDING: ' + esc(rec.label_name) : 'Not recording';
  $('elapsed').textContent = rec ? dur(rec.duration_s) : '-';
  $('count').textContent = rec ? rec.windows : '-';

  const lp = state.last_packet, feed = $('feed'), last = $('last');
  if (!lp) {
    feed.className = 'feed bad'; feed.textContent = 'No data received yet: is the ESP32/STM32 running?';
    last.className = 'bad'; last.textContent = 'none yet';
  } else if (lp.age_s > 5) {
    feed.className = 'feed bad'; feed.textContent = 'No data arriving (last ' + dur(lp.age_s) + ' ago)';
    last.className = 'bad'; last.textContent = dur(lp.age_s) + ' ago';
  } else {
    feed.className = 'feed ok';
    feed.textContent = 'Data arriving: ' + lp.rpm + ' RPM, ' + (lp.valid_channels.toString(2).split('1').length - 1) + ' of 4 channels OK';
    last.className = ''; last.textContent = lp.age_s.toFixed(1) + ' s ago';
  }

  const key = JSON.stringify([state.runs, state.summary]);
  if (key === runsKey) return;
  runsKey = key;
  const max = Math.max(1, ...Object.values(state.summary));
  $('summary').innerHTML = LABELS.map(([k, n]) =>
    '<div class="sum" style="--c:' + COLORS[k] + '">' + esc(n) + '<b>' + state.summary[k] + '</b>' +
    '<div class="bar"><i style="width:' + (100 * state.summary[k] / max) + '%"></i></div></div>').join('');
  $('all').className = 'btn' + (state.runs.length ? '' : ' off');
  if (!state.runs.length) {
    $('runs').innerHTML = '<div class="empty">No runs yet. Choose a condition and press Start recording.</div>';
    return;
  }
  $('runs').innerHTML = '<table><thead><tr><th>Label</th><th>Started</th><th>Duration</th><th>Windows</th>' +
    '<th>RPM</th><th>Note</th><th>CSV</th><th></th></tr></thead><tbody>' +
    state.runs.map(r => '<tr><td><span class="tag" style="--c:' + COLORS[r.label] + '">' + esc(r.label_name) + '</span></td>' +
      '<td>' + esc(r.started) + '</td><td class="num">' + dur(r.duration_s) + '</td><td class="num">' + r.windows + '</td>' +
      '<td>' + esc(r.rpm || '-') + '</td><td>' + esc(r.note) + '</td>' +
      '<td><a href="/export/' + esc(r.run_id) + '.csv">Download</a></td>' +
      '<td>' + (r.recording ? '<span class="live">recording</span>' :
        '<button class="del" data-id="' + esc(r.run_id) + '" data-text="' +
        esc(r.label_name + ', started ' + r.started + ', ' + r.windows + ' windows') + '">Delete</button>') + '</td></tr>').join('') +
    '</tbody></table>';
  for (const b of $('runs').querySelectorAll('.del')) b.onclick = () => del(b.dataset.id, b.dataset.text);
}

async function refresh() {
  try {
    const r = await fetch('/api/status', {cache:'no-store'});
    state = await r.json();
  } catch (e) {
    $('feed').className = 'feed bad'; $('feed').textContent = 'Recorder not reachable: is the script running?';
    return;
  }
  render();
}
render();
refresh();
setInterval(refresh, 1000);
</script>
</body>
</html>
'''.replace('__LABELS__', json.dumps(LABELS))


def main():
    parser = argparse.ArgumentParser(description='Record labelled training data from the COE node.')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--data-dir', default=str(HERE / 'training_data'))
    args = parser.parse_args()
    recorder = Recorder(args.data_dir)
    try:
        server = make_server(args.port, recorder)
    except OSError as error:
        raise SystemExit(f'Cannot use port {args.port} ({error}). '
                         'Is host_receiver.py or another recorder already running?')
    print(f'Open the UI:  http://localhost:{server.server_port}/', flush=True)
    print(f'ESP endpoint: POST http://<this laptop>:{server.server_port}/features', flush=True)
    print(f'Data folder:  {recorder.dir}  ({len(recorder.runs)} runs loaded)', flush=True)
    print(f'Beacon:       UDP {BEACON_PORT} "COE-HOST {server.server_port}" every {BEACON_PERIOD_S:g} s '
          f'from {", ".join(local_ipv4s()) or "no address"}', flush=True)
    stop_beacon = threading.Event()
    threading.Thread(target=run_beacon, args=(server.server_port, recorder, stop_beacon),
                     daemon=True, name='beacon').start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_beacon.set()
        if recorder.active:
            recorder.stop()
        server.server_close()


if __name__ == '__main__':
    main()
