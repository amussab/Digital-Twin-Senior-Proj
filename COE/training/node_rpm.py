"""Set or read the shaft speed the node assumes, over the STM32's ST-LINK USB cable.

    python node_rpm.py 3600        set 3600 RPM
    python node_rpm.py --query     ask what it is now
    python node_rpm.py 1800 --port COM5

The node has no tachometer: the number you type is only a setting, so it must match the VFD.
The STM32 applies it at the next window boundary (a window is at most 0.72 s) and answers with
one line starting with '@'. Exit code 0 means '@OK' (or an '@RPM' answer to --query).
The speed is not saved: after a reset the node is back at its power-up default (1800).
Standard library only (pyserial is used when installed). Windows.
"""
import argparse
import ctypes
import re
import subprocess
import sys
import time

BAUD = 115200
WAIT_S = 3.0
TRIES = 3  # a reply line can be lost when the node's log queue is full; the command is safe to repeat


def find_port():
    """COM port of the ST-LINK virtual COM port (USB vendor 0483), or None."""
    command = ("Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPDeviceID -match 'VID_0483' "
               "-and $_.Name -match '[(]COM[0-9]+[)]' } | ForEach-Object { $_.Name }")
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command', command],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        out = ''
    match = re.search(r'COM[0-9]+', out)
    if match:
        return match.group(0)
    try:  # fallback for PCs without PowerShell CIM
        out = subprocess.run(['wmic', 'path', 'Win32_PnPEntity', 'where', "PNPDeviceID like '%VID_0483%'",
                              'get', 'Name'], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r'COM[0-9]+', out)
    return match.group(0) if match else None


def build_command(rpm):
    return 'RPM?\n' if rpm is None else f'RPM {int(rpm)}\n'


def is_ok(reply, rpm):
    if reply is None:
        return False
    return reply.startswith('@RPM ') if rpm is None else reply.startswith('@OK RPM')


def pick_reply(lines):
    """First machine-readable line (starts with '@'), or None."""
    for line in lines:
        line = line.strip()
        if line.startswith('@'):
            return line
    return None


class _Win32Port:
    """Raw COM handle with non-blocking reads (a plain open() would wait forever for a byte)."""

    class _Timeouts(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulong) for n in ('ReadIntervalTimeout', 'ReadTotalTimeoutMultiplier',
                    'ReadTotalTimeoutConstant', 'WriteTotalTimeoutMultiplier', 'WriteTotalTimeoutConstant')]

    def __init__(self, name):
        k32 = ctypes.windll.kernel32
        k32.CreateFileW.restype = ctypes.c_void_p
        self.k32 = k32
        bs = chr(92)
        self.handle = k32.CreateFileW(bs * 2 + '.' + bs + name, 0xC0000000, 0, None, 3, 0, None)  # R+W, OPEN_EXISTING
        if self.handle in (None, ctypes.c_void_p(-1).value):
            raise OSError(f'cannot open {name}')
        timeouts = self._Timeouts(0xFFFFFFFF, 0, 0, 0, 1000)  # reads return at once
        k32.SetCommTimeouts(ctypes.c_void_p(self.handle), ctypes.byref(timeouts))

    def write(self, data):
        done = ctypes.c_ulong()
        self.k32.WriteFile(ctypes.c_void_p(self.handle), data, len(data), ctypes.byref(done), None)

    def read(self, count):
        buffer = ctypes.create_string_buffer(count)
        done = ctypes.c_ulong()
        self.k32.ReadFile(ctypes.c_void_p(self.handle), buffer, count, ctypes.byref(done), None)
        return buffer.raw[:done.value]

    def close(self):
        self.k32.CloseHandle(ctypes.c_void_p(self.handle))


class Port:
    """A tiny serial port: pyserial if present, else the Windows 'mode' command plus a file handle."""

    def __init__(self, name):
        self.name = name
        try:
            import serial  # type: ignore
            self.serial = serial.Serial(name, BAUD, timeout=0.2)
            self.raw = None
        except ImportError:
            self.serial = None
            subprocess.run(['mode', name, f'BAUD={BAUD}', 'PARITY=n', 'DATA=8', 'STOP=1'],
                           shell=True, capture_output=True, check=False)
            self.raw = _Win32Port(name)

    def write(self, text):
        data = text.encode('ascii')
        (self.serial or self.raw).write(data)

    def read_line(self, deadline):
        """Next line, or None at the deadline."""
        buffer = b''
        while time.monotonic() < deadline:
            if self.serial:
                chunk = self.serial.read(1)
            else:
                chunk = self.raw.read(1)  # returns at once on a COM port with no data
                if not chunk:
                    time.sleep(0.01)
            if not chunk:
                continue
            if chunk in (b'\n', b'\r'):
                if buffer:
                    return buffer.decode('ascii', 'replace')
                continue
            buffer += chunk
        return None

    def close(self):
        (self.serial or self.raw).close()


def send(port_name, rpm, wait_s=WAIT_S, tries=TRIES):
    """Send the command; return the '@' reply line, or None if there was none."""
    port = Port(port_name)
    try:
        for _ in range(tries):
            port.write(build_command(rpm))
            deadline = time.monotonic() + wait_s
            while True:
                line = port.read_line(deadline)
                if line is None:
                    break
                if line.startswith('@'):
                    return line
    finally:
        port.close()
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description='Set or read the node shaft speed (RPM).')
    parser.add_argument('rpm', nargs='?', type=int, help='shaft speed to set, 60..3600')
    parser.add_argument('--query', action='store_true', help='read the current speed instead')
    parser.add_argument('--port', help='COM port (default: find the ST-LINK port)')
    args = parser.parse_args(argv)
    if args.query == (args.rpm is not None):
        parser.error('give either an RPM value or --query')
    rpm = None if args.query else args.rpm
    port = args.port or find_port()
    if not port:
        print('ST-LINK COM port not found: is the STM32 USB cable plugged in?', file=sys.stderr)
        return 2
    try:
        reply = send(port, rpm)
    except OSError as error:
        print(f'Cannot use {port}: {error} (another program, for example a terminal, may have it open)',
              file=sys.stderr)
        return 2
    if reply is None:
        print(f'No reply from the node on {port}. Is the STM32 powered and running the LIVE firmware?',
              file=sys.stderr)
        return 3
    print(reply)
    return 0 if is_ok(reply, rpm) else 1


if __name__ == '__main__':
    sys.exit(main())
