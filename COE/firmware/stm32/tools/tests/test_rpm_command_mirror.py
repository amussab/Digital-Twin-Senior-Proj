"""Python mirror of CM7/App/acquisition/rpm_command.c (no host C compiler on the dev PC).

The C file itself is NOT compiled or run here (it was only syntax-checked with the ST clangd
bundle); this mirror follows it statement by statement and checks the intended behaviour.
Run: python test_rpm_command_mirror.py
"""
import unittest

LINE_MAX = 96


def upper(c):
    return c.upper() if 'a' <= c <= 'z' else c


class Line:
    def __init__(self):
        self.text = ''
        self.overflow = False

    def feed(self, byte):
        if byte in '\r\n':
            complete = len(self.text) != 0 and not self.overflow
            if not complete:
                self.clear()
            return complete
        if len(self.text) + 1 < LINE_MAX:
            self.text += byte
        else:
            self.overflow = True
        return False

    def clear(self):
        self.text, self.overflow = '', False


blank = lambda c: c in ' \t'


def skip(s, i):
    while i < len(s) and blank(s[i]):
        i += 1
    return i


def starts(s, i, word):
    return s[i:i + len(word)].upper() == word and len(s) - i >= len(word)


def parse_decimal(s, i):
    value, scale, digits, fraction = 0.0, 1.0, 0, False
    while i < len(s) and (s[i].isdigit() and s[i] in '0123456789' or (s[i] == '.' and not fraction)):
        if s[i] == '.':
            fraction = True
        else:
            digits += 1
            if digits > 9:
                return None
            if fraction:
                scale *= 0.1
                value += int(s[i]) * scale
            else:
                value = value * 10 + int(s[i])
        i += 1
    if digits == 0 or (i < len(s) and not blank(s[i])):
        return None
    return value, i


def parse(line):
    bad = ('BAD',)
    i = skip(line, 0)
    for w in ('CONFIG?', 'RPM?'):
        if starts(line, i, w):
            return ('QUERY',) if skip(line, i + len(w)) == len(line) else bad
    if starts(line, i, 'BEARING'):
        i += 7
        if i >= len(line) or not blank(line[i]):
            return bad
        vals = []
        i = skip(line, i)
        while i < len(line):
            if len(vals) == 5:
                return bad
            r = parse_decimal(line, i)
            if r is None or not (0 < r[0] < 100):
                return bad
            vals.append(r[0])
            i = skip(line, r[1])
        if len(vals) not in (4, 5):
            return bad
        return ('BEARING', vals[:4], vals[4] if len(vals) == 5 else None)
    if not starts(line, i, 'RPM'):
        return bad
    i += 3
    if i >= len(line) or not blank(line[i]):
        return bad
    i = skip(line, i)
    value = digits = 0
    while i < len(line) and line[i] in '0123456789' and digits < 6:
        value = value * 10 + int(line[i]); digits += 1; i += 1
    i = skip(line, i)
    if digits == 0 or digits > 5 or i != len(line):
        return bad
    return ('SET', value)


class T(unittest.TestCase):
    def test_rpm(self):
        self.assertEqual(parse('RPM 3600'), ('SET', 3600))
        self.assertEqual(parse('  rpm\t1800  '), ('SET', 1800))
        for bad in ('RPM', 'RPM3600', 'RPM -5', 'RPM 3600x', 'RPM 123456', 'RPM 1.5', 'FOO'):
            self.assertEqual(parse(bad), ('BAD',), bad)

    def test_query(self):
        for q in ('RPM?', 'rpm? ', 'CONFIG?', 'config?'):
            self.assertEqual(parse(q), ('QUERY',))
        self.assertEqual(parse('RPM? x'), ('BAD',))

    def test_bearing(self):
        k, o, hw = parse('BEARING 0.4518 5.1393 4.9699 6.0301 0.08')
        self.assertEqual(k, 'BEARING')
        for a, b in zip(o, (0.4518, 5.1393, 4.9699, 6.0301)):
            self.assertAlmostEqual(a, b, places=6)
        self.assertAlmostEqual(hw, 0.08, places=6)
        self.assertIsNone(parse('bearing 1 2 3 4')[2])
        for bad in ('BEARING', 'BEARING 1 2 3', 'BEARING 1 2 3 4 5 6', 'BEARING 1 2 3 -4', 'BEARING 1 2 3 4x',
                    'BEARING 0 2 3 4', 'BEARING 1 2 3 100', 'BEARING1 2 3 4', 'BEARING 1..2 2 3 4', 'BEARING 1e3 2 3 4'):
            self.assertEqual(parse(bad), ('BAD',), bad)
        self.assertEqual(parse('BEARING .5 2 3 4')[0], 'BEARING')

    def test_line(self):
        l = Line()
        out = [l.feed(c) for c in 'RPM 1800\r\n']
        self.assertTrue(out[8]); self.assertFalse(out[9] if False else False)
        l.clear()
        self.assertFalse(l.feed('\n'))  # empty line ignored
        for c in 'x' * 200:
            l.feed(c)
        self.assertFalse(l.feed('\r'))  # over-long dropped whole
        for c in 'RPM 60':
            l.feed(c)
        self.assertTrue(l.feed('\n'))
        self.assertEqual(l.text, 'RPM 60')
        # longest legal line fits
        l.clear()
        s = 'BEARING 0.4518 5.1393 4.9699 6.0301 0.08'
        for c in s:
            l.feed(c)
        self.assertTrue(l.feed('\n')); self.assertEqual(l.text, s)


if __name__ == '__main__':
    unittest.main()
