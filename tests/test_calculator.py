"""Calculator tool tests: arithmetic correctness and safety."""

from __future__ import annotations

import pytest

from app.tools.base import MalformedToolInputError, ToolExecutionError
from app.tools.calculator import CalculatorOutput, CalculatorTool

tool = CalculatorTool()


def calc(expression: str) -> float:
    output = tool.run({"expression": expression})
    assert isinstance(output, CalculatorOutput)
    return output.result


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def test_addition() -> None:
    assert calc("2 + 3") == 5
    assert calc("0 + 0") == 0


def test_subtraction() -> None:
    assert calc("10 - 4") == 6
    assert calc("4 - 10") == -6


def test_multiplication() -> None:
    assert calc("6 * 7") == 42
    assert calc("-3 * 5") == -15


def test_division() -> None:
    assert calc("10 / 4") == 2.5
    assert calc("100 / 10 / 5") == 2


def test_parentheses() -> None:
    assert calc("(2 + 3) * 4") == 20
    assert calc("2 + (3 * 4)") == 14
    assert calc("((1 + 2) * (3 + 4))") == 21


def test_decimals() -> None:
    assert calc("1.5 * 2") == 3
    assert calc("0.25 + 0.75") == 1
    assert calc(".5 * 4") == 2


def test_unary_operators() -> None:
    assert calc("-5 + 2") == -3
    assert calc("+5 + 2") == 7
    assert calc("2 * -3") == -6


def test_operator_precedence() -> None:
    assert calc("2 + 3 * 4") == 14
    assert calc("(2 + 3) * 4") == 20


def test_float_rounding_is_stable() -> None:
    assert calc("0.1 + 0.2") == 0.3
    assert calc("1 / 3") == pytest.approx(0.333333333333, rel=1e-9)


def test_output_contract() -> None:
    output = tool.run({"expression": "2 + 2"})
    assert output.model_dump() == {"expression": "2 + 2", "result": 4}


# ---------------------------------------------------------------------------
# Invalid expressions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression",
    ["2+", "(2 + 3", "2 + 3)", "* 5", "2 3", "..", "1.2.3", "()", "+"],
)
def test_invalid_expressions_raise_typed_error(expression: str) -> None:
    with pytest.raises(MalformedToolInputError):
        tool.run({"expression": expression})


def test_empty_expression_rejected_by_schema() -> None:
    with pytest.raises(MalformedToolInputError):
        tool.run({"expression": ""})


def test_expression_too_long_rejected() -> None:
    with pytest.raises(MalformedToolInputError):
        tool.run({"expression": "1+" * 500 + "1"})


def test_deeply_nested_expression_rejected() -> None:
    with pytest.raises(MalformedToolInputError):
        tool.run({"expression": "(" * 80 + "1" + ")" * 80})


# ---------------------------------------------------------------------------
# Unsafe input - no code execution possible
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",
        "os.system('ls')",
        "eval('2+2')",
        "2 if 1 else 3",
        "open('secrets.txt')",
        "1; print(2)",
        "2 ** 1000",
        "5 // 2",
        "5 % 2",
        "1e10",
        "lambda: 1",
        "a + b",
        "'string'",
        "[1, 2]",
        "2+2 # comment",
        "$(whoami)",
        "`id`",
        "../../etc/passwd",
    ],
)
def test_unsafe_expressions_are_rejected(expression: str) -> None:
    with pytest.raises(MalformedToolInputError):
        tool.run({"expression": expression})


def test_no_eval_in_source() -> None:
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "app" / "tools" / "calculator.py").read_text(
        encoding="utf-8"
    )
    assert "eval(" not in source
    assert "exec(" not in source


# ---------------------------------------------------------------------------
# Division by zero
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("expression", ["1 / 0", "1 / (2 - 2)", "5 / 0.0"])
def test_division_by_zero_fails_explicitly(expression: str) -> None:
    with pytest.raises(ToolExecutionError, match="division by zero"):
        tool.run({"expression": expression})
