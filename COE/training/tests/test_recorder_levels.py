"""Recorder health-level / bearing-id / position / baseline fields, coverage table and CSV columns.

Fake POSTs of node JSON (host_receiver schema) against an in-process recorder server, no node needed.
    python -I -m unittest COE/training/tests/test_recorder_levels.py
"""
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('training_recorder', HERE.parent / 'training_recorder.py')
rec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rec)


def post(url, body, headers=None):
    req = urllib.request.Request(url, json.dumps(body).encode(), {'Content-Type': 'application/json', **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def node_msg(seq, rpm=1800):
    return {'schema_version': 1, 'sequence': seq, 'window_id': seq, 'rpm': rpm, 'bearing_profile': 1,
            'valid_channels': 15, 'acceleration_unit': 'g', 'device_id': 'NODE1', 'session_id': '0123456789abcdef',
            'features': [0.01 * (i + 1) for i in range(32)]}


def send(base, seq, rpm=1800):
    return post(base + '/features', node_msg(seq, rpm), {'Idempotency-Key': f'NODE1-0123456789abcdef-{seq}'})


class RecorderLevels(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.recorder = rec.Recorder(self.tmp.name)
        self.server = rec.make_server(0, self.recorder, '127.0.0.1')
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def test_fields_csv_and_coverage(self):
        code, d = post(self.base + '/api/start', {'label': 'healthy', 'severity': 0, 'is_baseline': True, 'bearing_id': 'B-00'})
        self.assertEqual(code, 200, d)
        for i in range(1, 4):
            self.assertEqual(send(self.base, i)[0], 200)
        self.assertEqual(post(self.base + '/api/stop', {})[0], 200)
        code, d = post(self.base + '/api/start', {'label': 'inner_race', 'severity': 2, 'bearing_id': 'B-03', 'position': 'b2'})
        self.assertEqual(code, 200, d)
        for i in range(10, 16):
            send(self.base, i, 3600)
        post(self.base + '/api/stop', {})
        # validation
        self.assertEqual(post(self.base + '/api/start', {'label': 'ball', 'severity': 0})[0], 400)
        self.assertEqual(post(self.base + '/api/start', {'label': 'healthy', 'severity': 2})[0], 400)
        self.assertEqual(post(self.base + '/api/start', {'label': 'ball', 'severity': 1, 'is_baseline': True})[0], 400)
        self.assertEqual(post(self.base + '/api/start', {'label': 'ball', 'severity': 1, 'position': 'zz'})[0], 400)
        st = json.loads(urllib.request.urlopen(self.base + '/api/status').read())
        cov = {(c['condition'], c['severity'], c['rpm']): c for c in st['coverage']}
        self.assertEqual(cov[('baseline', 0, 1800)]['windows'], 3)
        self.assertAlmostEqual(cov[('baseline', 0, 1800)]['minutes'], 3 * 1200 / 1800 / 60, places=2)
        self.assertEqual(cov[('inner_race', 2, 3600)]['windows'], 6)
        text = urllib.request.urlopen(self.base + '/export/all.csv').read().decode()
        rows = list(csv.DictReader(io.StringIO(text)))
        self.assertEqual(len(rows), 9)
        header = list(rows[0].keys())
        self.assertEqual(header[:9], rec.CSV_COLUMNS[:9])
        self.assertEqual(header[-4:], ['severity', 'bearing_id', 'position', 'is_baseline'])   # appended at the END
        self.assertEqual(rows[0]['is_baseline'], '1')
        self.assertEqual((rows[-1]['severity'], rows[-1]['bearing_id'], rows[-1]['position']), ('2', 'B-03', 'b2'))
        # page serves
        self.assertIn(b'id="sev"', urllib.request.urlopen(self.base + '/').read())

    def test_old_meta_loads(self):
        d = Path(self.tmp.name)
        (d / 'run_20260101_000000.json').write_text(json.dumps(
            {'run_id': 'run_20260101_000000', 'label': 'cage', 'note': '', 'started_at': '2026-01-01T00:00:00+03:00',
             'stopped_at': '2026-01-01T00:01:00+03:00'}))
        (d / 'run_20260101_000000.jsonl').write_text(json.dumps({'received_at': 'x', 'message': node_msg(1)}) + '\n')
        r2 = rec.Recorder(self.tmp.name)
        m = r2.runs['run_20260101_000000']
        self.assertEqual((m['severity'], m['position'], m['is_baseline']), (None, 'both', False))
        self.assertEqual(r2.status()['coverage'][0]['severity'], None)


if __name__ == '__main__':
    unittest.main()
