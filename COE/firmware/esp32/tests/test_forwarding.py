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
                ('features', ctypes.c_float * 32), ('schema', ctypes.c_uint32),
                ('measured_rpm', ctypes.c_float), ('revolutions', ctypes.c_uint32),
                ('shaft_flags', ctypes.c_uint32), ('probes', ctypes.c_float * 8),
                ('stm_latency_us', ctypes.c_uint32), ('esp_latency_us', ctypes.c_uint32),
                ('wifi_latency_us', ctypes.c_uint32), ('sample_rate_millihz', ctypes.c_uint32),
                ('rpm_min', ctypes.c_float), ('rpm_max', ctypes.c_float), ('adc_gaps', ctypes.c_uint32),
                ('unsteady_windows', ctypes.c_uint32), ('stm_send_failures', ctypes.c_uint32),
                ('esp_rx_rejected', ctypes.c_uint32), ('esp_dropped', ctypes.c_uint32),
                ('wifi_link', ctypes.c_uint32), ('wifi_rssi_dbm', ctypes.c_int32)]


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

    def shaft(self):
        """Schema 2 packet with Keyphasor speed and both probes (synthetic test values)."""
        packet = self.packet()
        packet.schema, packet.measured_rpm, packet.revolutions, packet.shaft_flags = 2, 1749.93, 20, 0x1F
        packet.probes[:] = [1231.9, 52.25, 50.0, 70.5, 1041.4, 20.0, 12.0, 200.0]
        return packet

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

    def test_schema2_shaft_fields(self):
        packet = self.shaft()
        length, body = self.encode(packet)
        message = json.loads(body)
        self.assertEqual(validate_vector(message), ('coe-test', '1234567890abcdef', 1))
        self.assertEqual(message['schema_version'], 2)
        self.assertEqual(struct.pack('<f', message['measured_rpm']), struct.pack('<f', packet.measured_rpm))
        self.assertEqual((message['revolutions'], message['shaft_flags']), (20, 0x1F))
        names = [f'{p}_{n}' for p in ('prox1', 'prox3') for n in ('gap_um', 'pkpk_um', 'x1_um_pp', 'x1_phase_deg')]
        for name, value in zip(names, packet.probes):
            self.assertEqual(struct.pack('<f', message[name]), struct.pack('<f', value))
        self.assertEqual(self.post(message)[0], 200)
        # Schema 1 JSON has none of the shaft fields.
        self.assertNotIn('measured_rpm', json.loads(self.encode(self.packet())[1]))

    def test_schema3_latency_fields(self):
        packet = self.shaft()
        packet.schema, packet.stm_latency_us, packet.esp_latency_us, packet.wifi_latency_us = 3, 4321, 250, 38000
        packet.sample_rate_millihz = 29998500
        packet.rpm_min, packet.rpm_max, packet.adc_gaps, packet.unsteady_windows = 1749.5, 1750.25, 3, 7
        packet.stm_send_failures, packet.esp_rx_rejected, packet.esp_dropped = 1, 2, 4
        packet.wifi_link, packet.wifi_rssi_dbm = 1, -48
        length, body = self.encode(packet, sequence=50)
        message = json.loads(body)
        self.assertEqual(validate_vector(message), ('coe-test', '1234567890abcdef', 50))
        self.assertEqual(message['schema_version'], 3)
        uart = -(-(232 + 10) * 10 * 1000000 // 115200)  # wire time + idle timeout, rounded up
        self.assertEqual((message['stm_latency_us'], message['uart_latency_us'], message['esp_latency_us'],
                          message['wifi_latency_us'], message['node_latency_us']),
                         (4321, uart, 250, 38000, 4321 + uart + 250 + 38000))
        self.assertEqual(struct.pack('<f', message['measured_rpm']), struct.pack('<f', packet.measured_rpm))
        self.assertEqual((message['sample_rate_hz'], message['revolutions']), (29998.5, 20))
        self.assertIn('"sample_rate_hz":29998.500', body.decode())
        self.assertEqual([message[name] for name in ('rpm_min', 'rpm_max', 'adc_gaps', 'unsteady_windows',
                          'stm_send_failures', 'esp_rx_rejected', 'esp_dropped', 'wifi_link', 'wifi_rssi_dbm')],
                         [1749.5, 1750.25, 3, 7, 1, 2, 4, 'node', -48])
        # No Wi-Fi link: the signal is null.
        packet.wifi_link, packet.wifi_rssi_dbm = 0, 0
        unlinked = json.loads(self.encode(packet, sequence=51)[1])
        self.assertEqual((unlinked['wifi_link'], unlinked['wifi_rssi_dbm']), ('none', None))
        validate_vector(unlinked)
        for link, rssi in ((0, -40), (1, 1), (1, -128), (3, -40)):
            packet.wifi_link, packet.wifi_rssi_dbm = link, rssi
            self.assertEqual(self.encode(packet), (0, b''))
        packet.wifi_link, packet.wifi_rssi_dbm = 1, -48
        self.assertEqual(self.post(message)[0], 200)
        # Bearing profiles are 1..3.
        with self.assertRaises(ValueError):
            validate_vector(dict(message, bearing_profile=4))
        # The STM32 part is never 0 in schema 3; the JSON encoder refuses it.
        packet.stm_latency_us = 0
        self.assertEqual(self.encode(packet), (0, b''))
        packet.stm_latency_us, packet.sample_rate_millihz = 4321, 0
        self.assertEqual(self.encode(packet), (0, b''))
        # Schema 2 JSON has no latency fields.
        self.assertNotIn('stm_latency_us', json.loads(self.encode(self.shaft())[1]))
        for name, value in (('stm_latency_us', 0), ('node_latency_us', 1), ('esp_latency_us', -1),
                            ('wifi_latency_us', 1.5), ('uart_latency_us', 'x'), ('sample_rate_hz', 999.9),
                            ('sample_rate_hz', 200000.001), ('sample_rate_hz', 'fast'), ('sample_rate_hz', True),
                            ('rpm_min', 1751), ('rpm_min', 0), ('rpm_max', 8001), ('adc_gaps', -1),
                            ('esp_dropped', 1.0), ('wifi_link', 'wired'), ('wifi_rssi_dbm', None),
                            ('wifi_rssi_dbm', 3), ('wifi_rssi_dbm', -48.5)):
            with self.assertRaises(ValueError):
                validate_vector(dict(message, **{name: value}))
        missing = dict(message)
        del missing['wifi_latency_us']
        with self.assertRaises(ValueError):
            validate_vector(missing)

    def test_schema2_rules_on_the_host(self):
        _, body = self.encode(self.shaft(), sequence=40)
        original = json.loads(body)
        for name, value in (('shaft_flags', 0x3F), ('shaft_flags', 0x1D), ('shaft_flags', 0x0E),
                            ('measured_rpm', 0), ('revolutions', 0), ('prox1_x1_phase_deg', 360),
                            ('prox1_gap_um', -1), ('prox3_x1_um_pp', float('nan')), ('measured_rpm', 'fast')):
            changed = dict(original)
            changed[name] = value
            with self.assertRaises(ValueError):
                validate_vector(changed)
        missing = dict(original)
        del missing['prox3_pkpk_um']
        with self.assertRaises(ValueError):
            validate_vector(missing)
        no_speed = dict(original, shaft_flags=0x06, measured_rpm=0, revolutions=0,
                        prox1_x1_um_pp=0, prox1_x1_phase_deg=0, prox3_x1_um_pp=0, prox3_x1_phase_deg=0)
        validate_vector(no_speed)

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
        for name, value in (('features', [1, 2]), ('rpm', 4001), ('schema_version', 2), ('schema_version', 4),
                            ('valid_channels', 0), ('acceleration_unit', 'm/s2')):
            changed = dict(original)
            changed[name] = value
            variants.append(changed)
        for message in variants:
            self.assertEqual(self.post(message)[0], 400)
        self.assertEqual(self.post(original, identity='incorrect')[0], 400)


if __name__ == '__main__':
    unittest.main(verbosity=2)
