"""Set or read the node's runtime configuration over the STM32's ST-LINK USB cable.

    python node_config.py --rpm 3600
    python node_config.py --bearing 0.4518 5.1393 4.9699 6.0301 [0.08]    orders x shaft speed [+ half width]
    python node_config.py --query                                         prints the @CONFIG line
    python node_config.py --rpm 1800 --port COM5
    python node_config.py --selftest                                      reply parser only, no port

The node has no tachometer: the speed you type is only a setting, so it must match the VFD.
The STM32 applies a change at the next window boundary (a window is at most 0.72 s) and answers
with one line starting with '@' (@OK ..., @ERR ... <reason>, @CONFIG ...). Exit code 0 = every
command got @OK / @CONFIG. Bearing is sent before RPM when both are given: each is validated
against what is already pending, so give an RPM that fits the NEW bearing (or send them apart).
Nothing is saved on the node: after a reset it is back at 1800 RPM and the 12 Nov bearing.
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
TRIES = 3  # a reply line can be lost when the node's log queue is full; the commands are safe to repeat


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
    """Legacy: RPM command text (None = query)."""
    return 'CONFIG?\n' if rpm is None else f'RPM {int(rpm)}\n'


def bearing_command(orders, half_width=None):
    parts = [f'{float(x):.6g}' for x in orders]
    if half_width is not None:
        parts.append(f'{float(half_width):.6g}')
    return 'BEARING ' + ' '.join(parts) + '\n'


def _word(command):
    return command.split()[0].rstrip('?').upper()  # RPM, BEARING, CONFIG


def is_ok(reply, command):
    """True when reply is the success answer for command (an int/None legacy RPM or a command string)."""
    if reply is None:
        return False
    if not isinstance(command, str):
        command = build_command(command)
    if command.strip().endswith('?'):
        return reply.startswith('@CONFIG ') or reply.startswith('@RPM ')
    return reply.startswith(f'@OK {_word(command)}')


def pick_reply(lines, command=None):
    """First machine-readable line (starts with '@') that answers command (any '@' line if None)."""
    word = _word(command) if isinstance(command, str) else None
    for line in lines:
        line = line.strip()
        if not line.startswith('@'):
            continue
        if word is None or line.startswith(('@CONFIG', '@RPM ', f'@OK {word}', f'@ERR {word}',
                                            '@ERR CMD', '@ERR APPLY')):
            return line
    return None


def parse_config(line):
    """'@CONFIG rpm=1800 pending_rpm=none ftf=0.4518 ...' -> dict of strings, or None."""
    if not line or not line.startswith('@CONFIG'):
        return None
    return dict(re.findall(r'(\w+)=(\S+)', line))


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


def send(port_name, command, wait_s=WAIT_S, tries=TRIES):
    """Send one command (text, or a legacy int RPM / None for a query); return the '@' reply or None."""
    if not isinstance(command, str):
        command = build_command(command)
    if not command.endswith('\n'):
        command += '\n'
    port = Port(port_name)
    try:
        for _ in range(tries):
            port.write(command)
            deadline = time.monotonic() + wait_s
            while True:
                line = port.read_line(deadline)
                if line is None:
                    break
                if pick_reply([line], command):
                    return line.strip()
    finally:
        port.close()
    return None


def selftest():
    assert is_ok('@OK RPM 3600', 'RPM 3600\n') and not is_ok('@ERR RPM 99999 out of range 60..3600', 'RPM 99999')
    assert is_ok('@OK BEARING ftf=0.4518 bsf=5.1393 bpfo=4.9699 bpfi=6.0301 hw=0.0800', 'BEARING 1 2 3 4')
    assert not is_ok('@ERR BEARING ftf=1.0000 bsf=2.0000 bpfo=3.0000 bpfi=4.0000 hw=0.0800 bands overlap at 1800 RPM',
                     'BEARING 1 2 3 4')
    cfg = ('@CONFIG rpm=1800 pending_rpm=none ftf=0.4518 bsf=5.1393 bpfo=4.9699 bpfi=6.0301 hw=0.0800 '
           'pending_bearing=none fw=abc1234')
    assert is_ok(cfg, 'CONFIG?') and is_ok(cfg, None) and parse_config(cfg)['fw'] == 'abc1234'
    assert parse_config(cfg)['pending_rpm'] == 'none' and parse_config('@OK RPM 1') is None
    assert pick_reply(['  window 5 ...', '@OK RPM 1800', '@ERR BEARING x']) == '@OK RPM 1800'
    assert pick_reply(['@OK RPM 1', '@OK BEARING a'], 'BEARING 1 2 3 4') == '@OK BEARING a'
    assert pick_reply(['junk', '[1s] rate'], 'RPM 5') is None
    assert bearing_command([0.4518, 5.1393, 4.9699, 6.0301], 0.08) == 'BEARING 0.4518 5.1393 4.9699 6.0301 0.08\n'
    assert bearing_command([1, 2, 3, 4]) == 'BEARING 1 2 3 4\n'
    print('node_config selftest OK')


def main(argv=None, legacy_positional=False):
    parser = argparse.ArgumentParser(description='Set or read the node runtime configuration.')
    if legacy_positional:
        parser.add_argument('rpm_pos', nargs='?', type=int, help='shaft speed to set, 60..3600')
    parser.add_argument('--rpm', type=int, help='shaft speed to set, 60..3600')
    parser.add_argument('--bearing', nargs='+', type=float, metavar='X',
                        help='ftf bsf bpfo bpfi [half_width] (orders as multiples of shaft speed)')
    parser.add_argument('--query', action='store_true', help='read the configuration (CONFIG?)')
    parser.add_argument('--port', help='COM port (default: find the ST-LINK port)')
    parser.add_argument('--selftest', action='store_true', help='check the reply parser; no port needed')
    args = parser.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    rpm = args.rpm if args.rpm is not None else getattr(args, 'rpm_pos', None)
    if args.bearing is not None and len(args.bearing) not in (4, 5):
        parser.error('--bearing needs 4 or 5 numbers: ftf bsf bpfo bpfi [half_width]')
    if rpm is None and args.bearing is None and not args.query:
        parser.error('give --rpm, --bearing and/or --query')
    commands = []
    if args.bearing is not None:
        commands.append(bearing_command(args.bearing[:4], args.bearing[4] if len(args.bearing) == 5 else None))
    if rpm is not None:
        commands.append(f'RPM {int(rpm)}\n')
    if args.query:
        commands.append('CONFIG?\n')
    port = args.port or find_port()
    if not port:
        print('ST-LINK COM port not found: is the STM32 USB cable plugged in?', file=sys.stderr)
        return 2
    status = 0
    for command in commands:
        try:
            reply = send(port, command)
        except OSError as error:
            print(f'Cannot use {port}: {error} (another program, for example a terminal, may have it open)',
                  file=sys.stderr)
            return 2
        if reply is None:
            print(f'No reply to {command.strip()!r} on {port}. Is the STM32 powered and running the LIVE firmware?',
                  file=sys.stderr)
            return 3
        print(reply)
        if not is_ok(reply, command):
            status = 1
    return status


if __name__ == '__main__':
    sys.exit(main())
