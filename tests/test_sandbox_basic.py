"""Basic correctness tests for run_in_sandbox on well-formed inputs."""
from __future__ import annotations

import pytest

from infinity_forge.sandbox import run_in_sandbox


def test_identity_with_int():
    result = run_in_sandbox("def f(x): return x", 42)
    assert result["status"] == "ok", result
    assert result["output"] == 42
    assert result["error"] is None
    assert isinstance(result["runtime_ms"], int)
    assert result["runtime_ms"] >= 0


@pytest.mark.parametrize(
    "value",
    [
        0,
        -17,
        3.14,
        "hello",
        "",
        [1, 2, 3],
        [],
        {"a": 1, "b": [2, 3]},
        {},
        True,
        False,
        None,
    ],
)
def test_identity_various_json_types(value):
    result = run_in_sandbox("def f(x): return x", value)
    assert result["status"] == "ok", result
    assert result["output"] == value


def test_arithmetic():
    result = run_in_sandbox("def f(x): return x * 2 + 1", 10)
    assert result["status"] == "ok", result
    assert result["output"] == 21


def test_list_construction():
    result = run_in_sandbox("def f(x): return [x, x+1, x+2]", 5)
    assert result["status"] == "ok", result
    assert result["output"] == [5, 6, 7]


def test_allowed_builtins_sum_range():
    result = run_in_sandbox("def f(x): return sum(range(x))", 10)
    assert result["status"] == "ok", result
    assert result["output"] == 45  # 0+1+...+9


def test_dict_return():
    result = run_in_sandbox(
        "def f(x):\n    return {'value': x, 'double': x * 2}", 7
    )
    assert result["status"] == "ok", result
    assert result["output"] == {"value": 7, "double": 14}
