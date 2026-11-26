"""Unit tests for the /calc oracle (qwenputer/qpcalc.py).

Run: python3 -m pytest qwenputer/test_qpcalc.py   (or plain python3, below)
The kernel side is verified against the same expectations by
qwenputer/qp_serial_test.py booting the real OS in QEMU.
"""
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qpcalc import calc, CalcError  # noqa: E402

CASES = [
    ("0", 0),
    ("42", 42),
    ("2+3*4", 14),
    ("(2+3)*4", 20),
    ("123456789*987654321", 121932631112635269),
    ("100/7", 14),
    ("100%7", 2),
    ("-100/7", -14),          # idiv truncates toward zero
    ("-100%7", -2),
    ("100/-7", -14),
    ("-5*3+1", -14),
    ("--5", 5),
    ("-(3+4)", -7),
    ("2*(3+4*(5-2))", 30),
    ("9223372036854775807+1", -9223372036854775808),   # wrap
    ("-9223372036854775808/-1", -9223372036854775808),  # guarded idiv overflow
    ("-9223372036854775808%-1", 0),
    ("  12 +  30 ", 42),
    ("1000000*1000000*1000000", 1000000000000000000),
]

ERRORS = ["10/0", "10%0", "2+", "(3", "3)", "abc", "", "2 3", "/5"]


def test_cases():
    for expr, want in CASES:
        got = calc(expr)
        assert got == want, f"{expr!r}: got {got}, want {want}"


def test_errors():
    for expr in ERRORS:
        try:
            calc(expr)
        except CalcError:
            continue
        raise AssertionError(f"{expr!r} should have raised CalcError")


def test_fuzz_vs_python():
    rng = random.Random(20261125)
    for _ in range(500):
        a = rng.randint(-10**12, 10**12)
        b = rng.randint(1, 10**6) * rng.choice([1, -1])
        op = rng.choice(["+", "-", "*", "/", "%"])
        expr = f"{a}{op}{b}"
        if op == "+":
            want = a + b
        elif op == "-":
            want = a - b
        elif op == "*":
            want = a * b
        elif op == "/":
            want = abs(a) // abs(b) * (1 if (a >= 0) == (b >= 0) else -1)
        else:
            q = abs(a) // abs(b) * (1 if (a >= 0) == (b >= 0) else -1)
            want = a - q * b
        want = ((want + 2**63) % 2**64) - 2**63
        assert calc(expr) == want, expr


if __name__ == "__main__":
    test_cases()
    test_errors()
    test_fuzz_vs_python()
    print("qpcalc: all tests passed")
