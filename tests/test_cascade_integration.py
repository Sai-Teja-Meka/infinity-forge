"""Day 2 cascade integration checks.

Gate 4 only covers the cross-module sync test that guards the contract between
``sandbox._ALLOWED_BUILTIN_NAMES`` (parent-side, imported by the cascade) and
the ``_ALLOWED_BUILTINS`` dict literal embedded in ``sandbox._CHILD_SCRIPT``
(child-side, injected as ``__builtins__``). If these two diverge, the cascade
would accept names the child refuses or reject names the child accepts.
The end-to-end ``gate`` integration tests land at Gate 6.
"""
from __future__ import annotations

import ast

from infinity_forge import sandbox
from infinity_forge.cascade import gate


def test_child_script_builtins_match_exposed_allowlist():
    tree = ast.parse(sandbox._CHILD_SCRIPT)

    found_keys: set[str] | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_ALLOWED_BUILTINS":
                    value = node.value
                    assert isinstance(value, ast.Dict), (
                        "_ALLOWED_BUILTINS in _CHILD_SCRIPT must be a dict literal; "
                        f"got {type(value).__name__}"
                    )
                    keys: set[str] = set()
                    for k in value.keys:
                        assert isinstance(k, ast.Constant) and isinstance(k.value, str), (
                            "_ALLOWED_BUILTINS keys must be string literals; "
                            f"got {ast.dump(k) if k is not None else 'None (unpacking)'}"
                        )
                        keys.add(k.value)
                    found_keys = keys

    assert found_keys is not None, (
        "_ALLOWED_BUILTINS dict literal not found inside sandbox._CHILD_SCRIPT"
    )
    assert found_keys == set(sandbox._ALLOWED_BUILTIN_NAMES), (
        "child _ALLOWED_BUILTINS keys and parent _ALLOWED_BUILTIN_NAMES have drifted:\n"
        f"  only in child:  {sorted(found_keys - set(sandbox._ALLOWED_BUILTIN_NAMES))}\n"
        f"  only in parent: {sorted(set(sandbox._ALLOWED_BUILTIN_NAMES) - found_keys)}"
    )


def test_gate_accepted_candidate_computes_correctly():
    result = gate("def f(x):\n    return x * 2\n", 5)
    assert result["stage"] == "completed"
    assert result["accepted"] is True
    assert result["reason"] is None
    assert result["value"] == 10
    assert isinstance(result["duration_ms"], float)
    assert result["duration_ms"] > 0
    assert result["metadata"] == {"sandbox_status": "ok"}


def test_gate_accepted_candidate_times_out():
    source = (
        "def f(x):\n"
        "    while True:\n"
        "        x = x + 1\n"
        "    return x\n"
    )
    result = gate(source, 0, timeout_ms=200)
    assert result["stage"] == "sandbox"
    assert result["accepted"] is False
    assert "timeout" in result["reason"]
    assert result["metadata"]["sandbox_status"] == "timeout"
    assert isinstance(result["duration_ms"], float)


def test_gate_accepted_candidate_raises():
    result = gate("def f(x):\n    return 1 / x\n", 0)
    assert result["stage"] == "sandbox"
    assert result["accepted"] is False
    assert "ZeroDivision" in result["reason"]
    assert result["metadata"]["sandbox_status"] == "error"


def test_gate_layer_1_rejection_short_circuits_sandbox():
    result = gate("def f(x):\n    return x\n", 42)
    assert result["accepted"] is False
    assert result["stage"] == "layer_1"
    assert result["duration_ms"] is None
    assert result["value"] is None


def test_gate_layer_2_rejection_short_circuits_sandbox():
    result = gate("def f(x):\n    return open(x)\n", "whatever")
    assert result["accepted"] is False
    assert result["stage"] == "layer_2"
    assert result["duration_ms"] is None
    assert result["value"] is None


def test_gate_default_timeout_when_none_passed():
    # timeout_ms=None must not be forwarded to run_in_sandbox; the sandbox's
    # own 500ms default applies. If the kwargs gating were wrong (e.g., we
    # forwarded timeout_ms=None), the sandbox would divide None / 1000.0.
    result = gate("def f(x):\n    return x + 1\n", 7, timeout_ms=None)
    assert result["stage"] == "completed"
    assert result["accepted"] is True
    assert result["value"] == 8
