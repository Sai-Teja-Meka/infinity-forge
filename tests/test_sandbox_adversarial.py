"""Adversarial test battery for run_in_sandbox.

Every test asserts two things:

1. The sandbox returns a well-defined status envelope within 2 seconds of
   wall-clock time (the parent process regains control).
2. Whatever the hostile code tried to do — exhaust memory, escape via
   import, write to disk, open a socket, crash the interpreter — either
   fails cleanly with a reported error/crash/timeout status, or is
   contained so that no side effects leak onto the host.

If any test here fails, the Day-1 sandbox is broken and the rest of
∞ Forge cannot safely be built on top of it.
"""
from __future__ import annotations

import pathlib
import subprocess
import time
from typing import Any

import pytest

from infinity_forge.sandbox import run_in_sandbox

_WALL_CLOCK_CAP_S = 2.0


def _timed(
    source_code: str, input_value: Any = 0, **kwargs
) -> tuple[dict, float]:
    start = time.monotonic()
    result = run_in_sandbox(source_code, input_value, **kwargs)
    elapsed = time.monotonic() - start
    return result, elapsed


@pytest.fixture
def pwn_paths():
    """Remove the canary files before the test runs.

    We deliberately do NOT clean up afterwards — if a sandbox escape
    leaked a file, leaving it in place lets the post-suite manual
    verification (`ls /tmp/forge_*`) catch it even if the in-test
    assertion was somehow skipped or silenced.
    """
    pwned = pathlib.Path("/tmp/forge_pwned")
    pwned_file = pathlib.Path("/tmp/forge_pwned_file")
    pwned.unlink(missing_ok=True)
    pwned_file.unlink(missing_ok=True)
    return pwned, pwned_file


def test_infinite_loop_while_true():
    src = "def f(x):\n    while True:\n        pass\n    return x"
    result, elapsed = _timed(src, 0, timeout_ms=200)
    assert result["status"] == "timeout", result
    assert elapsed < _WALL_CLOCK_CAP_S


def test_infinite_loop_with_work():
    src = "def f(x):\n    y = 0\n    while True:\n        y += 1"
    result, elapsed = _timed(src, 0, timeout_ms=200)
    assert result["status"] == "timeout", result
    assert elapsed < _WALL_CLOCK_CAP_S


def test_deep_recursion():
    src = (
        "def f(x):\n"
        "    def g(n):\n"
        "        return g(n + 1)\n"
        "    return g(0)"
    )
    result, elapsed = _timed(src, 0)
    assert result["status"] == "error", result
    assert "Recursion" in result["error"] or "recursion" in result["error"], result
    assert elapsed < _WALL_CLOCK_CAP_S


def test_memory_exhaustion_list():
    src = "def f(x):\n    return [0] * (10**9)"
    result, elapsed = _timed(src, 0)
    assert result["status"] in ("error", "crash"), result
    if result["status"] == "error":
        err_lower = result["error"].lower()
        assert "memory" in err_lower or "memoryerror" in err_lower, result
    assert elapsed < _WALL_CLOCK_CAP_S


def test_memory_exhaustion_string():
    src = 'def f(x):\n    return "a" * (10**9)'
    result, elapsed = _timed(src, 0)
    assert result["status"] in ("error", "crash"), result
    assert elapsed < _WALL_CLOCK_CAP_S


def test_import_attempt_os():
    src = "def f(x):\n    import os\n    return os.getcwd()"
    result, elapsed = _timed(src, 0)
    assert result["status"] == "error", result
    assert (
        "NameError" in result["error"]
        or "ImportError" in result["error"]
        or "__import__" in result["error"]
    ), result
    assert elapsed < _WALL_CLOCK_CAP_S


def test_dunder_import():
    src = 'def f(x):\n    return __import__("os").getcwd()'
    result, elapsed = _timed(src, 0)
    assert result["status"] == "error", result
    assert elapsed < _WALL_CLOCK_CAP_S


def test_exec_malicious(pwn_paths):
    pwned, _ = pwn_paths
    src = (
        "def f(x):\n"
        "    exec(\"import os; os.system('touch /tmp/forge_pwned')\")\n"
        "    return x"
    )
    result, elapsed = _timed(src, 0)
    assert result["status"] == "error", result
    assert elapsed < _WALL_CLOCK_CAP_S
    assert pwned.exists() is False, "SANDBOX ESCAPE: /tmp/forge_pwned was created"


def test_fork_bomb():
    src = (
        "def f(x):\n"
        "    import os\n"
        "    while True:\n"
        "        os.fork()"
    )
    result, elapsed = _timed(src, 0)
    assert result["status"] == "error", result
    assert elapsed < _WALL_CLOCK_CAP_S


def test_file_write_attempt(pwn_paths):
    _, pwned_file = pwn_paths
    src = (
        "def f(x):\n"
        '    with open("/tmp/forge_pwned_file", "w") as fp:\n'
        '        fp.write("pwned")\n'
        "    return x"
    )
    result, elapsed = _timed(src, 0)
    assert result["status"] == "error", result
    assert elapsed < _WALL_CLOCK_CAP_S
    assert (
        pwned_file.exists() is False
    ), "SANDBOX ESCAPE: /tmp/forge_pwned_file was created"


def test_file_read_attempt():
    src = (
        "def f(x):\n"
        '    with open("/etc/passwd", "r") as fp:\n'
        "        return fp.read()"
    )
    result, elapsed = _timed(src, 0)
    assert result["status"] == "error", result
    assert elapsed < _WALL_CLOCK_CAP_S


def test_network_attempt():
    src = (
        "def f(x):\n"
        "    import socket\n"
        "    s = socket.socket()\n"
        '    s.connect(("8.8.8.8", 53))\n'
        "    return x"
    )
    result, elapsed = _timed(src, 0)
    assert result["status"] == "error", result
    assert elapsed < _WALL_CLOCK_CAP_S


def test_crash_via_ctypes():
    src = (
        "def f(x):\n"
        "    import ctypes\n"
        "    ctypes.string_at(0)\n"
        "    return x"
    )
    result, elapsed = _timed(src, 0)
    assert result["status"] == "error", result
    assert elapsed < _WALL_CLOCK_CAP_S


def test_oversized_output():
    src = 'def f(x):\n    return "a" * 100000'
    result, elapsed = _timed(src, 0)
    assert result["status"] == "error", result
    assert result["error"] == "output_too_large", result
    assert elapsed < _WALL_CLOCK_CAP_S


def test_non_json_output():
    src = (
        "def f(x):\n"
        "    import datetime\n"
        "    return datetime.datetime.now()"
    )
    result, elapsed = _timed(src, 0)
    assert result["status"] == "error", result
    assert elapsed < _WALL_CLOCK_CAP_S


def test_multiple_functions():
    src = "def f(x):\n    return x\n\ndef g(x):\n    return x + 1"
    result, elapsed = _timed(src, 0)
    assert result["status"] == "bad_source", result
    assert result["runtime_ms"] == 0
    assert elapsed < 0.1


def test_syntax_error():
    src = "def f(x: return x"
    result, elapsed = _timed(src, 0)
    assert result["status"] == "bad_source", result
    assert result["runtime_ms"] == 0
    assert elapsed < 0.1


def test_no_function_definition():
    src = "x = 5"
    result, elapsed = _timed(src, 0)
    assert result["status"] == "bad_source", result
    assert result["runtime_ms"] == 0
    assert elapsed < 0.1


def test_non_json_input():
    result, elapsed = _timed("def f(x): return x", {1, 2, 3})
    assert result["status"] == "bad_input", result
    assert result["runtime_ms"] == 0
    assert elapsed < 0.1


def _pgrep_python3_count() -> int:
    proc = subprocess.run(
        ["pgrep", "-c", "python3"],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return int(proc.stdout.strip() or "0")
    except ValueError:
        return 0


# Ten timeouts at 200ms each plus overhead — comfortably under 30s, which
# we set to override the project-wide 3s default.
@pytest.mark.timeout(30)
def test_process_cleanup_on_timeout():
    src = "def f(x):\n    while True:\n        pass\n    return x"
    time.sleep(0.2)  # let any lingering child from earlier tests get reaped
    before = _pgrep_python3_count()
    for _ in range(10):
        result, _ = _timed(src, 0, timeout_ms=200)
        assert result["status"] == "timeout", result
    time.sleep(0.3)  # give the kernel a moment to reap
    after = _pgrep_python3_count()
    # Tolerate ±2 for unrelated host-level python3 processes coming/going.
    assert (
        after <= before + 2
    ), f"python3 process count grew from {before} to {after}; possible zombies"


# Realistic per-call cost is ~60-120ms on WSL2 (Python startup dominates),
# so 100 runs is ~6-12s. 90s override leaves ample headroom.
@pytest.mark.timeout(90)
def test_stress_100_safe_executions():
    src = "def f(x): return x * 2"
    start = time.monotonic()
    for i in range(100):
        result = run_in_sandbox(src, i)
        assert result["status"] == "ok", f"iteration {i}: {result}"
        assert result["output"] == i * 2
    elapsed = time.monotonic() - start
    assert elapsed < 60, f"100 safe executions took {elapsed:.2f}s (cap 60s)"
