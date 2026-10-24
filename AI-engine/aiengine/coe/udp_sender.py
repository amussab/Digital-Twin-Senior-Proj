"""Send 152-byte COE payloads to the backend UDP ingest (one datagram = one payload, no framing).

The backend (src/DigitalTwin.Backend/Ingest/IngestServices.cs, default port 5005) accepts only
private/loopback senders [spec C5] and treats any other datagram length as a reject.
"""
from __future__ import annotations

import socket
import time
from typing import Iterable, Optional

from .. import payload as payload_mod
from .pipeline import WindowResult


def parse_hostport(s: str, default_port: int = 5005) -> tuple[str, int]:
    host, _, port = s.partition(":")
    return host or "127.0.0.1", int(port) if port else default_port


class UdpSender:
    def __init__(self, host: str = "127.0.0.1", port: int = 5005):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sent = 0

    def send(self, raw: bytes) -> None:
        if len(raw) != payload_mod.PAYLOAD_SIZE:
            raise ValueError(f"payload must be {payload_mod.PAYLOAD_SIZE} bytes, got {len(raw)}")
        self.sock.sendto(raw, self.addr)
        self.sent += 1

    def close(self) -> None:
        self.sock.close()


def stream(windows: Iterable[WindowResult], sender: UdpSender, rate_hz: float = 0.0,
           max_windows: Optional[int] = None) -> int:
    """Send valid windows; `rate_hz` windows/s pacing (0 = as fast as produced)."""
    period = 1.0 / rate_hz if rate_hz > 0 else 0.0
    nxt = time.perf_counter()
    n = 0
    for w in windows:
        if not w.valid:
            continue
        if period:
            nxt += period
            delay = nxt - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        sender.send(w.payload)
        n += 1
        if max_windows is not None and n >= max_windows:
            break
    return n
