"""Arithmetic the agent may use in replies, checked so it can't launder numbers.

The draft validator only accepts figures that appear in a retrieved passage. That
rules out answers like "the merit scholarship saves you INR 18,600" (15% of
1,24,000), which no document writes down. These two tools compute such figures in
code, and their results become allowed figures for the draft.

The catch: a calculator could turn an injected number into a "computed" one
(10,000 + 0 = 10,000). So every operand must itself be a documented figure:
- any number with 2+ digits must appear in a passage retrieved in this run;
- single digits and 100 (for percentages) are allowed as plain constants;
- a date must appear in a passage as written there, e.g. "10 April 2027".
"""

from __future__ import annotations

import ast
import operator
import re
from datetime import date, timedelta

from lead_assistant.guardrails import numbers_in

MAX_EXPRESSION = 200
_FREE_CONSTANTS = {"100"}
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


class CalculationError(ValueError):
    """The expression is invalid or uses an undocumented operand."""


def _normalise(expression: str) -> str:
    e = re.sub(r"(?<=\d),(?=\d)", "", expression)          # 1,24,000 -> 124000
    e = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", e)     # 15% -> (15/100)
    return e.replace("×", "*").replace("÷", "/").replace("−", "-")


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise CalculationError("division by zero")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_evaluate(node.operand)
    raise CalculationError("only numbers, + - * /, % and parentheses are allowed")


def format_inr(value: float) -> str:
    """Indian digit grouping, as the documents write amounts: 146320 -> 1,46,320."""
    neg, value = value < 0, round(abs(value), 2)
    whole, frac = divmod(value, 1)
    s = str(int(whole))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = re.sub(r"(\d)(?=(\d{2})+$)", r"\1,", head)
        s = f"{head},{tail}"
    if frac:
        s += f"{frac:.2f}"[1:]
    return ("-" if neg else "") + s


def calculate(expression: str, source_text: str) -> str:
    """Evaluate `expression` if every operand is documented in `source_text`.
    Returns the result with Indian digit grouping."""
    if len(expression) > MAX_EXPRESSION:
        raise CalculationError(f"expression longer than {MAX_EXPRESSION} characters")
    normalised = _normalise(expression)
    try:
        tree = ast.parse(normalised, mode="eval")
    except SyntaxError:
        raise CalculationError(f"not a valid arithmetic expression: {expression!r}")
    documented = numbers_in(source_text) | _FREE_CONSTANTS
    operands = {str(n.value).removesuffix(".0") for n in ast.walk(tree) if isinstance(n, ast.Constant)}
    undocumented = sorted(o for o in operands if len(o.replace(".", "")) >= 2 and o not in documented)
    if undocumented:
        raise CalculationError(f"operands not found in the retrieved passages: {undocumented}. "
                               "Search for the document that states them first.")
    return format_inr(_evaluate(tree))


def shift_date(start: date, days: int, source_text: str) -> str:
    """`start` plus `days` (negative = before), if both are documented."""
    written = f"{start.day} {start:%B} {start.year}"
    if written not in source_text:
        raise CalculationError(f"date {written!r} does not appear in the retrieved passages")
    if abs(days) >= 10 and str(abs(days)) not in numbers_in(source_text):
        raise CalculationError(f"{abs(days)} days does not appear in the retrieved passages")
    result = start + timedelta(days=days)
    return f"{result.day} {result:%B} {result.year}"
