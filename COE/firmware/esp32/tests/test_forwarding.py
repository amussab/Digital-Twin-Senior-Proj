"""Exercise the actual C JSON encoder/decoder and example HTTP host on Linux GCC.

The ESP-IDF application and physical Wi-Fi/UART/LED still need a board build/test.
"""
import ctypes
import http.client
import json
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import threading
import unittest
from host_receiver import make_server, validate_vector

PROJECT = Path(__file__).resolve().parents[1]
BUILD = PROJECT / 'tests' / 'build'


class Packet(ctypes.Structure):
    """Mirror the portable, verified FeaturePacket C layout for test inputs."""
    _fields_ = [('window_id', ctypes.c_uint32), ('rpm', ctypes.c_uint32),
                ('bearing_profile', ctypes.c_uint32), ('valid_channels', ctypes.c_uint32),
                ('features', ctypes.c_float * 32)]


class ForwardingTests(unittest.TestCase):
    """Check actual serialization, independent Float32 round trips and HTTP acceptance."""

    @classmethod
    def setUpClass(cls):
        BUILD.mkdir(exist_ok=True)
        memory_check = BUILD / 'json_memory_checks'
        subprocess.run(['gcc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-pedantic',
                        '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                        '-I', str(PROJECT / 'main'), '-I', str(PROJECT / 'main/payload'),
                        str(PROJECT / 'main/payload/feature_json.c'),
                        str(PROJECT / 'tests/json_memory_checks.c'), '-o', str(memory_check)], check=True)
        subprocess.run([str(memory_check)], check=True)
        library = BUILD / 'feature_json.so'
        subprocess.run(['gcc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-pedantic',
                        '-shared', '-fPIC', '-I', str(PROJECT / 'main'),
                        '-I', str(PROJECT / 'main/payload'),
                        str(PROJECT / 'main/payload/feature_json.c'),
                        str(PROJECT / 'main/feature_packet.c'), '-o', str(library)], check=True)
        cls.library = ctypes.CDLL(str(library))
        cls.library.feature_packet_to_json.argtypes = [ctypes.POINTER(Packet), ctypes.c_char_p,
            ctypes.c_uint64, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t]
        cls.library.feature_packet_to_json.restype = ctypes.c_size_t
        cls.library.feature_packet_decode.argtypes = [ctypes.c_void_p, ctypes.POINTER(Packet)]
        cls.library.feature_packet_decode.restype = ctypes.c_bool
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name) / 'received.jsonl'
        cls.server = make_server('127.0.0.1', 0, cls.output)
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join()
        cls.temporary.cleanup()

    def packet(self):
        """Use distinctive synthetic Float32 values; these are test data, not board results."""
        return Packet(3, 1750, 1, 15, (ctypes.c_float * 32)(*[i / 10 for i in range(32)]))

    def encode(self, packet, capacity=2048, device=b'coe-test', sequence=1):
        buffer = ctypes.create_string_buffer(capacity)
        length = self.library.feature_packet_to_json(ctypes.byref(packet), device,
                                                    0x1234567890ABCDEF, sequence, buffer, capacity)
        return length, buffer.value

    def post(self, message, identity=None):
        """Exercise a real loopback HTTP request; no credentials or outside hosts are used."""
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=3)
        if identity is None:
            identity = f"{message['device_id']}-{message['session_id']}-{message['sequence']}"
        connection.request('POST', '/features', json.dumps(message),
                           {'Content-Type': 'application/json', 'Idempotency-Key': identity})
        response = connection.getresponse()
        result = response.status, json.loads(response.read())
        connection.close()
        return result

    def test_feature_order_and_metadata(self):
        packet = self.packet()
        length, body = self.encode(packet)
        self.assertEqual(length, len(body))
        message = json.loads(body)
        self.assertEqual(validate_vector(message), ('coe-test', '1234567890abcdef', 1))
        self.assertEqual(message['window_id'], 3)
        self.assertEqual(message['rpm'], 1750)
        for actual, received in zip(packet.features, message['features']):
            self.assertEqual(struct.pack('<f', actual), struct.pack('<f', received))

    def test_float32_roundtrip(self):
        rng = random.Random(42)
        packet = self.packet()
        for _ in range(100):
            for index in range(32):
                packet.features[index] = struct.unpack('<f', struct.pack('<I', rng.randrange(0x7F800000)))[0]
            length, body = self.encode(packet)
            self.assertGreater(length, 0)
            for actual, received in zip(packet.features, json.loads(body)['features']):
                self.assertEqual(struct.pack('<f', actual), struct.pack('<f', received))

    def test_buffer_bounds_and_invalid_input(self):
        packet = self.packet()
        self.assertEqual(self.encode(packet, capacity=32), (0, b''))
        self.assertEqual(self.encode(packet, device=b'bad"id'), (0, b''))
        self.assertEqual(self.encode(packet, sequence=0), (0, b''))
        for value in (float('nan'), float('inf'), -1.0):
            packet.features[0] = value
            self.assertEqual(self.encode(packet), (0, b''))

    def test_valid_http_save_retry_and_conflict(self):
        _, body = self.encode(self.packet(), sequence=27)
        message = json.loads(body)
        self.assertEqual(self.post(message), (200, {'accepted': True, 'duplicate': False}))
        self.assertEqual(self.post(message), (200, {'accepted': True, 'duplicate': True}))
        rows = [json.loads(line) for line in self.output.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(sum(row['sequence'] == 27 for row in rows), 1)
        message['features'][0] = 7
        self.assertEqual(self.post(message)[0], 409)

    def test_http_rejects_invalid_payload(self):
        _, body = self.encode(self.packet(), sequence=28)
        original = json.loads(body)
        variants = []
        for name, value in (('features', [1, 2]), ('rpm', 5000), ('schema_version', 2),
                            ('valid_channels', 0), ('acceleration_unit', 'm/s2')):
            changed = dict(original)
            changed[name] = value
            variants.append(changed)
        for message in variants:
            self.assertEqual(self.post(message)[0], 400)
        self.assertEqual(self.post(original, identity='incorrect')[0], 400)


if __name__ == '__main__':
    unittest.main(verbosity=2)
