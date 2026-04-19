"""Acceptance cases for the Day 2 cascade layers."""
from __future__ import annotations

from radon.complexity import cc_visit

from infinity_forge.cascade import evaluate, layer_1_structure, layer_2_safety


def _assert_accepted(source: str) -> dict:
    accepted, reason, metadata = layer_1_structure(source)
    assert accepted is True, f"expected accepted, got reason={reason!r}"
    assert reason is None
    assert isinstance(metadata, dict)
    return metadata


def _assert_layer2_accepted(source: str) -> dict:
    accepted, reason, metadata = layer_2_safety(source)
    assert accepted is True, f"expected layer_2 accepted, got reason={reason!r}"
    assert reason is None
    assert isinstance(metadata, dict)
    return metadata


def test_simple_pure_function():
    _assert_accepted("def f(x):\n    return x + 1\n")


def test_allowed_builtins_usage():
    _assert_accepted("def f(x):\n    return abs(sum(x))\n")


def test_local_variable_and_loop():
    source = (
        "def f(x):\n"
        "    total = 0\n"
        "    for item in x:\n"
        "        total = total + item\n"
        "    return total\n"
    )
    _assert_accepted(source)


def test_comprehension_with_filter():
    _assert_accepted("def f(x):\n    return [y for y in x if y > 0]\n")


def test_boundary_exactly_30_lines():
    body = "\n".join(f"    a{i} = x + {i}" for i in range(28))
    source = f"def f(x):\n{body}\n    return a0\n"
    assert len(source.splitlines()) == 30, "precondition: source is exactly 30 lines"
    metadata = _assert_accepted(source)
    assert metadata["lines"] == 30


def test_boundary_exactly_complexity_10():
    ifs = "\n".join(f"    if x == {i}: return {i}" for i in range(9))
    source = f"def f(x):\n{ifs}\n    return 9\n"
    measured = cc_visit(source)[0].complexity
    assert measured == 10, f"precondition: radon complexity pinned at 10, got {measured}"
    metadata = _assert_accepted(source)
    assert metadata["complexity"] == 10


def test_layer2_accept_tuple_unpacking_assign():
    source = (
        "def f(x):\n"
        "    a, b = x\n"
        "    return a + b\n"
    )
    meta = _assert_layer2_accepted(source)
    assert "a" in meta["locals"] and "b" in meta["locals"]


def test_layer2_accept_with_as_binding():
    source = (
        "def f(x):\n"
        "    with x as y:\n"
        "        return y + 1\n"
    )
    meta = _assert_layer2_accepted(source)
    assert "y" in meta["locals"]


def test_layer2_accept_except_as_binding():
    source = (
        "def f(x):\n"
        "    try:\n"
        "        y = x + 1\n"
        "    except x as e:\n"
        "        y = 0\n"
        "    return y\n"
    )
    meta = _assert_layer2_accepted(source)
    assert "e" in meta["locals"] and "y" in meta["locals"]


def test_layer2_accept_walrus_in_comprehension():
    source = "def f(x):\n    return [y for z in x if (y := z + 1) > 0]\n"
    meta = _assert_layer2_accepted(source)
    assert "y" in meta["locals"] and "z" in meta["locals"]


def test_layer2_accept_nested_def():
    source = (
        "def f(x):\n"
        "    def helper(y):\n"
        "        return y + 1\n"
        "    return helper(x) + x\n"
    )
    meta = _assert_layer2_accepted(source)
    assert "helper" in meta["locals"] and "y" in meta["locals"]


def test_layer2_accept_allowed_builtins():
    source = "def f(x):\n    return sorted(filter(None, map(abs, x)))\n"
    _assert_layer2_accepted(source)


_RESULT_FIELDS = {"accepted", "stage", "reason", "value", "duration_ms", "metadata"}


def _assert_result_shape(result) -> None:
    assert isinstance(result, dict)
    assert set(result.keys()) == _RESULT_FIELDS, f"unexpected Result keys: {set(result.keys())}"
    assert isinstance(result["accepted"], bool)
    assert isinstance(result["stage"], str)
    assert result["reason"] is None or isinstance(result["reason"], str)
    assert result["duration_ms"] is None or isinstance(result["duration_ms"], float)
    assert isinstance(result["metadata"], dict)


def test_evaluate_accept_simple_function_full_shape():
    result = evaluate("def f(x):\n    return x + 1\n")
    _assert_result_shape(result)
    assert result["accepted"] is True
    assert result["stage"] == "layer_2"
    assert result["reason"] is None
    assert result["value"] is None
    assert result["duration_ms"] is None
    # Combined metadata has Layer 1 keys and Layer 2 keys
    assert "lines" in result["metadata"]
    assert "complexity" in result["metadata"]
    assert "param_name" in result["metadata"]
    assert "locals" in result["metadata"]


def test_evaluate_accept_combined_metadata_preserves_both_layers():
    source = (
        "def f(x):\n"
        "    a, b = x\n"
        "    return a + b\n"
    )
    result = evaluate(source)
    _assert_result_shape(result)
    assert result["accepted"] is True
    assert result["stage"] == "layer_2"
    assert result["metadata"]["param_name"] == "x"
    assert result["metadata"]["complexity"] == 1
    assert "a" in result["metadata"]["locals"]
    assert "b" in result["metadata"]["locals"]


def test_evaluate_accept_value_and_duration_remain_none():
    result = evaluate("def f(x):\n    return abs(x)\n")
    assert result["accepted"] is True
    assert result["value"] is None
    assert result["duration_ms"] is None
