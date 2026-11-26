"""Exact signed 64-bit integer calculator — Python oracle for the kernel's /calc.

Mirrors 02-baremetal-os/qwenputer.asm exactly: grammar
    expr   := term { ('+'|'-') term }
    term   := factor { ('*'|'/'|'%') factor }
    factor := ('-'|'+') factor | '(' expr ')' | number
All arithmetic wraps to signed 64-bit (two's complement), division truncates
toward zero (x86 idiv semantics), x % -1 == 0, INT64_MIN / -1 == INT64_MIN.
Raises CalcError on bad input or divide by zero.
"""
from __future__ import annotations

MASK = (1 << 64) - 1
MIN = -(1 << 63)
MAX = (1 << 63) - 1


def wrap(v: int) -> int:
    v &= MASK
    return v - (1 << 64) if v > MAX else v


class CalcError(Exception):
    pass


class _Parser:
    def __init__(self, text: str):
        self.s = text
        self.i = 0

    def skip(self):
        while self.i < len(self.s) and self.s[self.i] in " \t":
            self.i += 1

    def peek(self) -> str:
        self.skip()
        return self.s[self.i] if self.i < len(self.s) else ""

    def expr(self) -> int:
        v = self.term()
        while True:
            c = self.peek()
            if c == "+":
                self.i += 1
                v = wrap(v + self.term())
            elif c == "-":
                self.i += 1
                v = wrap(v - self.term())
            else:
                return v

    def term(self) -> int:
        v = self.factor()
        while True:
            c = self.peek()
            if c == "*":
                self.i += 1
                v = wrap(v * self.factor())
            elif c == "/":
                self.i += 1
                d = self.factor()
                if d == 0:
                    raise CalcError("divide by zero")
                if v == MIN and d == -1:
                    v = MIN
                else:
                    # truncating division, exactly like idiv:
                    q = abs(v) // abs(d) * (1 if (v >= 0) == (d >= 0) else -1)
                    v = wrap(q)
            elif c == "%":
                self.i += 1
                d = self.factor()
                if d == 0:
                    raise CalcError("divide by zero")
                if d == -1:
                    v = 0
                else:
                    q = abs(v) // abs(d) * (1 if (v >= 0) == (d >= 0) else -1)
                    v = wrap(v - q * d)
            else:
                return v

    def factor(self) -> int:
        c = self.peek()
        if c == "-":
            self.i += 1
            return wrap(-self.factor())
        if c == "+":
            self.i += 1
            return self.factor()
        if c == "(":
            self.i += 1
            v = self.expr()
            if self.peek() != ")":
                raise CalcError("missing )")
            self.i += 1
            return v
        if c.isdigit():
            start = self.i
            while self.i < len(self.s) and self.s[self.i].isdigit():
                self.i += 1
            return wrap(int(self.s[start:self.i]))
        raise CalcError(f"unexpected character: {c!r}")


def calc(text: str) -> int:
    p = _Parser(text)
    v = p.expr()
    p.skip()
    if p.i != len(p.s):
        raise CalcError("trailing characters")
    return v
