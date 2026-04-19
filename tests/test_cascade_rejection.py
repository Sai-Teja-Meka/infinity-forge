"""Layer 1 and Layer 2 rejection cases — one test per rejection path."""
from __future__ import annotations

from radon.complexity import cc_visit

from infinity_forge.cascade import evaluate, layer_1_structure, layer_2_safety


def _assert_rejected(source: str, substring: str) -> str:
    accepted, reason, metadata = layer_1_structure(source)
    assert accepted is False
    assert reason is not None
    assert reason.startswith("layer_1: "), f"reason missing layer_1 prefix: {reason!r}"
    assert substring in reason, f"expected {substring!r} in {reason!r}"
    assert isinstance(metadata, dict)
    return reason


def _assert_layer2_rejected(source: str, substring: str) -> str:
    accepted, reason, metadata = layer_2_safety(source)
    assert accepted is False
    assert reason is not None
    assert reason.startswith("layer_2: "), f"reason missing layer_2 prefix: {reason!r}"
    assert substring in reason, f"expected {substring!r} in {reason!r}"
    assert isinstance(metadata, dict)
    return reason


def test_reject_syntax_error():
    _assert_rejected("def f(x:\n    return x\n", "syntax")


def test_reject_multiple_top_level_statements():
    _assert_rejected(
        "def f(x):\n    return x + 1\n\ndef g(x):\n    return x + 2\n",
        "multiple top-level",
    )


def test_reject_not_a_function():
    _assert_rejected("x = 1\n", "not a function")


def test_reject_async_function():
    _assert_rejected("async def f(x):\n    return x + 1\n", "async")


def test_reject_zero_args():
    _assert_rejected("def f():\n    return 1\n", "exactly one positional")


def test_reject_two_args():
    _assert_rejected("def f(x, y):\n    return x + y\n", "exactly one positional")


def test_reject_varargs():
    _assert_rejected("def f(*args):\n    return sum(args)\n", "*args")


def test_reject_kwargs():
    _assert_rejected("def f(**kwargs):\n    return len(kwargs)\n", "**kwargs")


def test_reject_keyword_only():
    _assert_rejected("def f(*, x):\n    return x + 1\n", "keyword-only")


def test_reject_default_value():
    _assert_rejected("def f(x=0):\n    return x + 1\n", "default")


def test_reject_trivial_pass():
    _assert_rejected("def f(x):\n    pass\n", "trivial pass")


def test_reject_trivial_identity_return():
    _assert_rejected("def f(x):\n    return x\n", "identity")


def test_reject_over_length_31_lines():
    body = "\n".join(f"    a{i} = x + {i}" for i in range(29))
    source = f"def f(x):\n{body}\n    return a0\n"
    assert len(source.splitlines()) == 31, "precondition: source is exactly 31 lines"
    _assert_rejected(source, "exceeds 30 lines")


def test_reject_over_complexity_11_ifs():
    ifs = "\n".join(f"    if x == {i}: return {i}" for i in range(11))
    source = f"def f(x):\n{ifs}\n    return 11\n"
    measured = cc_visit(source)[0].complexity
    assert measured > 10, f"precondition: radon complexity should exceed 10, got {measured}"
    _assert_rejected(source, "complexity")


def test_layer2_reject_import_in_function():
    source = "def f(x):\n    import os\n    return x\n"
    _assert_layer2_rejected(source, "import statement")


def test_layer2_reject_dunder_import_call():
    source = "def f(x):\n    return __import__('os')\n"
    _assert_layer2_rejected(source, "__import__")


def test_layer2_reject_dunder_attribute_access():
    source = "def f(x):\n    return x.__class__\n"
    _assert_layer2_rejected(source, "dunder attribute")


def test_layer2_reject_forbidden_builtin():
    source = "def f(x):\n    return eval(x)\n"
    _assert_layer2_rejected(source, "forbidden builtin")


def test_layer2_reject_open():
    source = "def f(x):\n    return open(x)\n"
    _assert_layer2_rejected(source, "forbidden builtin")


def test_layer2_reject_global():
    source = "def f(x):\n    global y\n    return x + 1\n"
    _assert_layer2_rejected(source, "global")


def test_layer2_reject_yield():
    source = "def f(x):\n    yield x + 1\n"
    _assert_layer2_rejected(source, "yield")


def test_layer2_reject_undefined_name():
    source = "def f(x):\n    return undefined_var + x\n"
    _assert_layer2_rejected(source, "undefined name")


def test_layer2_reject_del_statement():
    source = "def f(x):\n    y = x + 1\n    del y\n    return 0\n"
    _assert_layer2_rejected(source, "del statement")


def test_evaluate_stage_is_layer_1_on_layer_1_rejection():
    # trivial identity return -> Layer 1 rejection
    result = evaluate("def f(x):\n    return x\n")
    assert result["accepted"] is False
    assert result["stage"] == "layer_1"
    assert result["reason"] is not None and result["reason"].startswith("layer_1: ")
    assert result["value"] is None
    assert result["duration_ms"] is None


def test_evaluate_stage_is_layer_1_on_syntax_error():
    result = evaluate("def f(x:\n    return x\n")
    assert result["accepted"] is False
    assert result["stage"] == "layer_1"
    assert "syntax" in result["reason"]


def test_evaluate_stage_is_layer_2_on_layer_2_rejection():
    # Layer 1 passes (well-formed single-arg non-trivial function); Layer 2 rejects 'open'
    result = evaluate("def f(x):\n    return open(x)\n")
    assert result["accepted"] is False
    assert result["stage"] == "layer_2"
    assert result["reason"] is not None and result["reason"].startswith("layer_2: ")
    assert "forbidden builtin" in result["reason"]
    assert result["value"] is None
    assert result["duration_ms"] is None


def test_evaluate_stage_is_layer_2_on_import_rejection():
    result = evaluate("def f(x):\n    import os\n    return x + 1\n")
    assert result["accepted"] is False
    assert result["stage"] == "layer_2"
    assert "import statement" in result["reason"]
