"""Deterministic calculator tool with a safe expression evaluator.

NEVER uses Python's eval or exec, and never runs arbitrary code: a
hand-written tokenizer and recursive-descent parser accepts ONLY numbers,
+, -, *, /, parentheses, and whitespace. Anything else (identifiers,
attributes, imports, quotes) is rejected before evaluation.

Supported: addition, subtraction, multiplication, division, parentheses,
decimals, unary +/-. Division by zero fails explicitly.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from app.tools.base import (
    MalformedToolInputError,
    Tool,
    ToolExecutionError,
)

__all__ = ["CalculatorInput", "CalculatorOutput", "CalculatorTool"]

_MAX_EXPRESSION_CHARS = 500
_MAX_TOKENS = 200
_MAX_DEPTH = 64


class CalculatorInput(BaseModel):
    """Arguments for the calculator tool."""

    expression: str = Field(min_length=1, max_length=_MAX_EXPRESSION_CHARS)


class CalculatorOutput(BaseModel):
    """Deterministic result of evaluating an expression."""

    expression: str
    result: float


def _tokenize(expression: str) -> list[str]:
    """Split into numbers and single-character operators, rejecting anything
    that is not part of arithmetic syntax."""
    tokens: list[str] = []
    i = 0
    length = len(expression)
    while i < length:
        char = expression[i]
        if char.isspace():
            i += 1
            continue
        if char.isdigit() or char == ".":
            start = i
            dots = 0
            while i < length and (expression[i].isdigit() or expression[i] == "."):
                if expression[i] == ".":
                    dots += 1
                i += 1
            token = expression[start:i]
            if dots > 1 or token == ".":
                raise MalformedToolInputError(
                    f"invalid number {token!r} in expression",
                    tool_name="calculator",
                )
            tokens.append(token)
            continue
        if char in "+-*/()":
            tokens.append(char)
            i += 1
            continue
        raise MalformedToolInputError(
            f"unsupported character {char!r} - calculator accepts only "
            "numbers, +, -, *, /, parentheses, and whitespace",
            tool_name="calculator",
        )
    if len(tokens) > _MAX_TOKENS:
        raise MalformedToolInputError(
            f"expression too complex ({len(tokens)} tokens, max {_MAX_TOKENS})",
            tool_name="calculator",
        )
    return tokens


class _Parser:
    """Recursive-descent parser: expr -> term -> factor -> primary."""

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self._pos = 0
        self._depth = 0

    def parse(self) -> float:
        if not self._tokens:
            raise MalformedToolInputError(
                "expression is empty", tool_name="calculator"
            )
        value = self._expr()
        if self._pos != len(self._tokens):
            leftover = " ".join(self._tokens[self._pos :])
            raise MalformedToolInputError(
                f"unexpected trailing input: {leftover!r}",
                tool_name="calculator",
            )
        return value

    def _peek(self) -> str | None:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _next(self) -> str:
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    def _expr(self) -> float:
        self._enter()
        value = self._term()
        while self._peek() in ("+", "-"):
            operator = self._next()
            right = self._term()
            value = value + right if operator == "+" else value - right
        self._exit()
        return value

    def _term(self) -> float:
        value = self._factor()
        while self._peek() in ("*", "/"):
            operator = self._next()
            right = self._factor()
            if operator == "*":
                value = value * right
            else:
                if right == 0:
                    raise ToolExecutionError(
                        "division by zero", tool_name="calculator"
                    )
                value = value / right
        return value

    def _factor(self) -> float:
        token = self._peek()
        if token == "+":
            self._next()
            return self._factor()
        if token == "-":
            self._next()
            return -self._factor()
        return self._primary()

    def _primary(self) -> float:
        token = self._peek()
        if token is None:
            raise MalformedToolInputError(
                "expression ended unexpectedly", tool_name="calculator"
            )
        if token == "(":
            self._next()
            self._enter()
            value = self._expr()
            if self._peek() != ")":
                raise MalformedToolInputError(
                    "missing closing parenthesis", tool_name="calculator"
                )
            self._next()
            self._exit()
            return value
        if token in "*/)" or not _is_number(token):
            raise MalformedToolInputError(
                f"unexpected token {token!r}", tool_name="calculator"
            )
        self._next()
        return float(token)

    def _enter(self) -> None:
        self._depth += 1
        if self._depth > _MAX_DEPTH:
            raise MalformedToolInputError(
                f"expression nested too deeply (max {_MAX_DEPTH})",
                tool_name="calculator",
            )

    def _exit(self) -> None:
        self._depth -= 1


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def evaluate(expression: str) -> float:
    """Safely evaluate an arithmetic expression to a float.

    Raises ``MalformedToolInputError`` for invalid/unsafe syntax and
    ``ToolExecutionError`` for division by zero.
    """
    tokens = _tokenize(expression)
    value = _Parser(tokens).parse()
    if value != value or value in (float("inf"), float("-inf")):  # NaN/inf guard
        raise ToolExecutionError(
            "expression produced a non-finite result", tool_name="calculator"
        )
    return float(f"{value:.12g}")


class CalculatorTool(Tool):
    """Safe arithmetic: +, -, *, /, parentheses, decimals. No code execution."""

    name: ClassVar[str] = "calculator"
    description: ClassVar[str] = (
        "Evaluate an arithmetic expression (+, -, *, /, parentheses, decimals). "
        "Returns the numeric result. Does not execute code."
    )
    input_schema: ClassVar[type[BaseModel]] = CalculatorInput
    output_schema: ClassVar[type[BaseModel]] = CalculatorOutput
    timeout_s: ClassVar[float] = 5.0
    max_retries: ClassVar[int] = 0
    retry_backoff_s: ClassVar[float] = 0.0

    def execute(self, arguments: BaseModel) -> BaseModel:
        assert isinstance(arguments, CalculatorInput)
        result = evaluate(arguments.expression)
        return CalculatorOutput(expression=arguments.expression, result=result)
