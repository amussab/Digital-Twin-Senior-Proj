"""Example HTTP host for COE vectors: validate, save JSONL and acknowledge.

Run only when ready to receive: python host_receiver.py --bind 0.0.0.0 --port 8000
This is a development host, with a bounded in-memory deduplication history.
"""
import argparse
from collections import OrderedDict
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import math
from pathlib import Path
import re


PROBES = ('prox1', 'prox3')
PROBE_VALUES = ('gap_um', 'pkpk_um', 'x1_um_pp', 'x1_phase_deg')
SHAFT_FIELDS = [f'{probe}_{value}' for probe in PROBES for value in PROBE_VALUES]


def number(message, name):
    """A finite JSON number (not a bool)."""
    value = message.get(name)
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return value


def validate_shaft(message):
    """Schema 2: Keyphasor speed and RK4 probe values, consistent with shaft_flags.

    shaft_flags bits: 0 speed measured, 1 PROX 1 valid, 2 PROX 3 valid,
    3 PROX 1 1X valid, 4 PROX 3 1X valid. A value that is not valid is 0.
    """
    for name in ('revolutions', 'shaft_flags'):
        if type(message.get(name)) is not int:
            raise ValueError(f'{name} must be an integer')
    flags = message['shaft_flags']
    if not 0 <= flags <= 0x1F:
        raise ValueError('Invalid shaft flags')
    speed = bool(flags & 1)
    measured = number(message, 'measured_rpm')
    if speed and not (60 <= measured <= 4000 and message['revolutions'] >= 1):
        raise ValueError('Invalid measured speed')
    if not speed and (measured != 0 or message['revolutions'] != 0):
        raise ValueError('Speed values without a measured speed')
    for index, probe in enumerate(PROBES):
        valid, x1 = bool(flags & (2 << index)), bool(flags & (8 << index))
        if x1 and not (valid and speed):
            raise ValueError(f'{probe} 1X needs a valid probe and a measured speed')
        for position, value_name in enumerate(PROBE_VALUES):
            name = f'{probe}_{value_name}'
            value = number(message, name)
            if valid if position < 2 else x1:
                if value < 0 or (value_name == 'x1_phase_deg' and value >= 360):
                    raise ValueError(f'Invalid {name}')
            elif value != 0:
                raise ValueError(f'{name} must be 0 when not valid')


def validate_vector(message):
    """Reject wrong metadata/order length and unusable numeric values."""
    if not isinstance(message, dict):
        raise ValueError('Expected a JSON object')
    for name in ('schema_version', 'sequence', 'window_id', 'rpm', 'bearing_profile', 'valid_channels'):
        if type(message.get(name)) is not int:
            raise ValueError(f'{name} must be an integer')
    if message['schema_version'] not in (1, 2) or not 60 <= message['rpm'] <= 4000:
        raise ValueError('Unsupported schema or speed')
    if message['bearing_profile'] not in (1, 2, 3) or message.get('acceleration_unit') != 'g':
        raise ValueError('Unsupported bearing profile or unit')
    if not 1 <= message['window_id'] <= 0xFFFFFFFF or not 1 <= message['sequence'] <= 0xFFFFFFFF:
        raise ValueError('Invalid window or delivery sequence')
    mask = message['valid_channels']
    if not 0 <= mask <= 15:
        raise ValueError('Invalid channel mask')
    if not isinstance(message.get('device_id'), str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,32}', message['device_id']):
        raise ValueError('Invalid device identifier')
    if not isinstance(message.get('session_id'), str) or not re.fullmatch(r'[0-9a-f]{16}', message['session_id']):
        raise ValueError('Invalid session identifier')
    features = message.get('features')
    if not isinstance(features, list) or len(features) != 32:
        raise ValueError('Expected exactly 32 ordered features')
    for index, value in enumerate(features):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 3.4028234663852886e38:
            raise ValueError(f'Invalid feature {index}')
        if not mask & (1 << (index // 8)) and value != 0:
            raise ValueError('An invalid channel must have zero placeholders')
    if message['schema_version'] == 2:
        validate_shaft(message)
    return (message['device_id'], message['session_id'], message['sequence'])


class FeatureHandler(BaseHTTPRequestHandler):
    """Accept POST /features and return a 2xx only after the row is saved."""

    def reply(self, status, data):
        """Send one bounded JSON response and close the HTTP/1.0 connection."""
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        """Validate the schema, suppress retries and append one accepted vector."""
        if self.path != '/features':
            self.reply(404, {'accepted': False})
            return
        self.connection.settimeout(5)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 4096:
                raise ValueError('Expected a body of 1-4096 bytes')
            if self.headers.get_content_type() != 'application/json':
                raise ValueError('Expected application/json')
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError('Incomplete body')
            message = json.loads(raw)
            key = validate_vector(message)
            identity = f'{key[0]}-{key[1]}-{key[2]}'
            if self.headers.get('Idempotency-Key') != identity:
                raise ValueError('Delivery identity/header mismatch')
        except (KeyError, TypeError, ValueError, OverflowError, TimeoutError) as error:
            self.reply(400, {'accepted': False, 'error': str(error)})
            return
        canonical = json.dumps(message, sort_keys=True, allow_nan=False)
        if key in self.server.seen:
            if self.server.seen[key] != canonical:
                self.reply(409, {'accepted': False, 'error': 'Identity reused with different data'})
            else:
                self.reply(200, {'accepted': True, 'duplicate': True})
            return
        try:
            with self.server.output.open('a', encoding='utf-8') as output:
                output.write(canonical + '\n')
        except OSError:
            self.reply(500, {'accepted': False, 'error': 'Could not save vector'})
            return
        self.server.seen[key] = canonical
        if len(self.server.seen) > 10000:
            self.server.seen.popitem(last=False)
        if not getattr(self.server, 'quiet', False):
            speed = f", Keyphasor {message['measured_rpm']:.2f} RPM" if message.get('shaft_flags', 0) & 1 else ''
            print(f"Saved window={message['window_id']}, RPM={message['rpm']}{speed}, features=32", flush=True)
        self.reply(200, {'accepted': True, 'duplicate': False})


def make_server(bind, port, output):
    """Create a host whose deduplication state is owned by this server instance."""
    server = HTTPServer((bind, port), FeatureHandler)
    server.output = Path(output)
    server.seen = OrderedDict()
    return server


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bind', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--output', default=str(Path(__file__).with_name('received_features.jsonl')))
    args = parser.parse_args()
    server = make_server(args.bind, args.port, args.output)
    print(f'Ready: POST http://{args.bind}:{server.server_port}/features; output={server.output}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
