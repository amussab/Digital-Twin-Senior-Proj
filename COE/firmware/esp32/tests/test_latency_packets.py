"""Check the actual timing encoder/parser against independent Python CRC frames."""
import ctypes as c
from pathlib import Path
import struct
import subprocess
import tempfile
import zlib

ESP = Path(__file__).resolve().parents[1]

class Stream(c.Structure):
    _fields_ = [('bytes', c.c_uint8 * 36), ('count', c.c_size_t)]

class Ack(c.Structure):
    _fields_ = [(name, c.c_uint32) for name in ('measurement_id', 'window_id', 'rpm',
                'bearing_profile', 'esp_total_us', 'esp_queue_us', 'esp_network_us')]

def frame(magic, values):
    """Use standard-library CRC/packing independently of the firmware encoder."""
    body = magic + struct.pack('<' + 'I' * len(values), *values)
    return body + struct.pack('<I', zlib.crc32(body))

with tempfile.TemporaryDirectory() as temporary:
    library = Path(temporary) / 'latency.so'
    subprocess.run(['gcc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-pedantic',
                    '-shared', '-fPIC', '-I', str(ESP / 'main'), str(ESP / 'main/transport/latency_packet.c'),
                    str(ESP / 'main/feature_packet.c'),
                    '-o', str(library)], check=True)
    lib = c.CDLL(str(library))
    lib.latency_request_push.argtypes = [c.POINTER(Stream), c.c_uint8, c.POINTER(c.c_uint32)]
    lib.latency_request_push.restype = c.c_bool
    lib.latency_ack_push.argtypes = [c.POINTER(Stream), c.c_uint8, c.POINTER(Ack)]
    lib.latency_ack_push.restype = c.c_bool
    lib.latency_request_encode.argtypes = [c.c_uint32, c.c_void_p]
    lib.latency_ack_encode.argtypes = [c.POINTER(Ack), c.c_void_p]

    def accepted(data, kind):
        """Feed arbitrary stream bytes and retain only accepted frames."""
        stream, output = Stream(), c.c_uint32() if kind == 'request' else Ack()
        results = []
        push = lib.latency_request_push if kind == 'request' else lib.latency_ack_push
        for byte in data:
            if push(c.byref(stream), byte, c.byref(output)):
                results.append(output.value if kind == 'request' else tuple(getattr(output, n) for n, _ in Ack._fields_))
            assert stream.count <= (12 if kind == 'request' else 36)
        return results

    count = 0
    for number in (1, 2, 0x31454F43, 0xFFFFFFFF):
        request = frame(b'COT1', [number])
        buffer = c.create_string_buffer(12)
        lib.latency_request_encode(number, buffer)
        assert buffer.raw == request
        assert accepted(request, 'request') == [number]
        count += 2
    assert not accepted(frame(b'COT1', [0]), 'request')
    count += 1
    fields = (7, 3, 3600, 1, 123456, 3456, 120000)
    ack = frame(b'COA1', fields)
    buffer = c.create_string_buffer(36)
    lib.latency_ack_encode(c.byref(Ack(*fields)), buffer)
    assert buffer.raw == ack
    assert accepted(ack, 'ack') == [fields]
    count += 2
    for kind, valid, result in (('request', request, number), ('ack', ack, fields)):
        for cut in range(len(valid)):
            assert accepted(b'noiseCO' + valid[:cut] + valid, kind) == [result]
            count += 1
        for index in range(len(valid)):
            for bit in range(8):
                damaged = bytearray(valid)
                damaged[index] ^= 1 << bit
                assert not accepted(damaged, kind)
                assert accepted(damaged + valid, kind) == [result]
                count += 2
    for index, wrong in ((0, 0), (1, 0), (2, 1000), (3, 3), (4, 123455), (5, 3457), (6, 0xFFFFFFFF)):
        values = list(fields)
        values[index] = wrong
        assert not accepted(frame(b'COA1', values), 'ack')
        count += 1

    class FeatureStream(c.Structure):
        _fields_ = [('bytes', c.c_uint8 * 160), ('count', c.c_size_t), ('rejected_packets', c.c_uint32)]
    class Feature(c.Structure):
        _fields_ = [(n, c.c_uint32) for n in ('window_id', 'rpm', 'bearing_profile', 'valid_channels')] + [('features', c.c_float * 32)]
    class Transport(c.Structure):
        _fields_ = [('features', FeatureStream), ('requests', Stream),
                    ('request_bytes_left', c.c_uint), ('measurement_id', c.c_uint32)]
    lib.latency_feature_push.argtypes = [c.POINTER(Transport), c.c_uint8, c.POINTER(Feature), c.POINTER(c.c_uint32)]
    lib.latency_feature_push.restype = c.c_bool
    lib.latency_feature_reset.argtypes = [c.POINTER(Transport)]
    body = b'COE1' + struct.pack('<HHIIIII', 1, 32, 3, 3600, 1, 15, 1) + struct.pack('<32f', *[i / 10 for i in range(32)])
    vector = body + struct.pack('<I', zlib.crc32(body))

    def decode_transport(data):
        """Call the same portable demultiplexer used by the ESP UART task."""
        state, value, identity = Transport(), Feature(), c.c_uint32()
        results = []
        for byte in data:
            if lib.latency_feature_push(c.byref(state), byte, c.byref(value), c.byref(identity)):
                results.append((identity.value, value.window_id, value.rpm, bytes(value.features)))
        return results

    expected_values = bytes(Feature.from_buffer_copy(struct.pack('<IIII', 3, 3600, 1, 15) + body[28:]).features)
    for identity in (1, 7, 0x31454F43, 0xFFFFFFFF):
        tagged = frame(b'COT1', [identity]) + vector
        assert decode_transport(b'noiseCO' + tagged + vector) == [(identity, 3, 3600, expected_values), (0, 3, 3600, expected_values)]
        count += 1
    corrupted_request = bytearray(frame(b'COT1', [7]))
    corrupted_request[8] ^= 1
    assert decode_transport(corrupted_request + vector)[0][0] == 0
    count += 1
    # A CRC-valid timing frame embedded in finite feature floats is data, not control.
    for identity in range(1, 1000):
        embedded = frame(b'COT1', [identity])
        values = struct.unpack('<3f', embedded)
        if all(0 <= value < 3e38 for value in values):
            modified = body[:28] + embedded + body[40:]
            packet = modified + struct.pack('<I', zlib.crc32(modified))
            assert decode_transport(packet)[0][0] == 0
            count += 1
            break
    else:
        raise AssertionError('No finite embedded timing fixture found')
    state = Transport()
    state.measurement_id = 99
    state.request_bytes_left = 3
    state.features.rejected_packets = 10
    lib.latency_feature_reset(c.byref(state))
    assert state.measurement_id == 0 and state.request_bytes_left == 0 and state.features.rejected_packets == 10
    count += 1
    print(f'PASS {count} independent timing-frame checks (CRC, corruption, metadata, boundaries, recovery).')
