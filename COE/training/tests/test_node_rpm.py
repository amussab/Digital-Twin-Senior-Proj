"""Host tests: node_rpm helpers, and a Python mirror of the firmware command parser.

The C parser (firmware/stm32/CM7/App/acquisition/rpm_command.c) is pure C, but this PC has no
host C compiler, so its rules are mirrored here and kept in step by hand. Run:
    python -I -m unittest COE/training/tests/test_node_rpm.py   (or: python tests/test_node_rpm.py)
"""
import importlib.util
from pathlib import Path
import re
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('node_rpm', HERE.parent / 'node_rpm.py')
node_rpm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(node_rpm)


def parse(line):
    """Mirror of rpm_command_parse: ('SET', n), ('QUERY', 0) or ('BAD', 0)."""
    if re.fullmatch(r'[ \t]*[Rr][Pp][Mm]\?[ \t]*', line):
        return ('QUERY', 0)
    m = re.fullmatch(r'[ \t]*[Rr][Pp][Mm][ \t]+([0-9]{1,5})[ \t]*', line)
    return ('SET', int(m.group(1))) if m else ('BAD', 0)


def feed(chunks, limit=24):
    """Mirror of rpm_line_feed: lines completed by \r or \n; empty/too long ones dropped."""
    lines, cur, over = [], '', False
    for c in chunks:
        if c in '\r\n':
            if cur and not over:
                lines.append(cur)
            cur, over = '', False
        elif len(cur) + 1 < limit:
            cur += c
        else:
            over = True
    return lines


class Parser(unittest.TestCase):
    def test_set(self):
        self.assertEqual(parse('RPM 3600'), ('SET', 3600))
        self.assertEqual(parse('  rpm\t1800  '), ('SET', 1800))
        self.assertEqual(parse('RPM 060'), ('SET', 60))

    def test_query(self):
        self.assertEqual(parse('RPM?'), ('QUERY', 0))
        self.assertEqual(parse('rpm? '), ('QUERY', 0))

    def test_bad(self):
        for text in ('', 'RPM', 'RPM ', 'RPM -5', 'RPM 36x0', 'RPM 123456', 'RPM 3600 1', 'RPM3600', 'HELLO'):
            self.assertEqual(parse(text), ('BAD', 0), text)

    def test_lines(self):
        self.assertEqual(feed('RPM 3600\r\nRPM?\n\n'), ['RPM 3600', 'RPM?'])
        self.assertEqual(feed('x' * 40 + '\nRPM 1\n'), ['RPM 1'])


class Host(unittest.TestCase):
    def test_command_text(self):
        self.assertEqual(node_rpm.build_command(3600), 'RPM 3600\n')
        # The query became CONFIG? (it reports rpm and the bearing); the firmware still accepts RPM?.
        self.assertEqual(node_rpm.build_command(None), 'CONFIG?\n')

    def test_reply(self):
        lines = ['  window 5 (... RPM) ...', '@OK RPM 3600', 'other']
        self.assertEqual(node_rpm.pick_reply(lines), '@OK RPM 3600')
        self.assertTrue(node_rpm.is_ok('@OK RPM 3600', 3600))
        self.assertFalse(node_rpm.is_ok('@ERR RPM 99 out of range 60..3600', 99))
        self.assertTrue(node_rpm.is_ok('@RPM 1800 PENDING none', None))
        self.assertFalse(node_rpm.is_ok(None, 5))


if __name__ == '__main__':
    unittest.main()
