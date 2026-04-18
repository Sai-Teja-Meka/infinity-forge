"""Subprocess-based sandbox for executing candidate Python functions.

This module provides :func:`run_in_sandbox`, the only public entry point of
Day 1 of ∞ Forge. It runs a single-function Python source string in an
isolated ``python3`` child process with kernel-enforced resource limits
(CPU, address space, file descriptors, process count, file size), a
wall-clock timeout, and a restricted set of built-ins. Communication with
the child is JSON over stdin/stdout; the child's source code is *never*
passed on the command line.

The sandbox is deliberately stdlib-only. It does NOT perform AST-level
banned-node checking beyond "exactly one top-level ``FunctionDef``"; that
static-safety pass is Layer 1 of the Day 2 cascade. The sandbox's job is
to ensure that *whatever* ends up running, the host system and the parent
process cannot be harmed and the parent regains control within a bounded
wall-clock budget.
"""
from __future__ import annotations

import ast
import json
import resource
import shutil
import subprocess
import time
from typing import Any

_PYTHON3 = shutil.which("python3")
if _PYTHON3 is None:
    raise RuntimeError(
        "infinity_forge.sandbox requires the 'python3' executable on PATH; "
        "none was found at module import time."
    )

_MAX_OUTPUT_BYTES = 10 * 1024
_CPU_SECONDS_HARD_CAP = 1
_FD_LIMIT = 16
_NPROC_LIMIT = 1
_FSIZE_LIMIT = 0
_POST_KILL_WAIT_S = 0.5

_CHILD_SCRIPT = r"""
import sys
import json
import io

# Preserve the real stdout fd so function-level print() cannot corrupt the
# JSON result envelope the parent reads from stdout.
_REAL_STDOUT = sys.stdout

_ALLOWED_BUILTINS = {
    "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
    "range": range, "sorted": sorted, "reversed": reversed,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "all": all, "any": any, "round": round,
    "int": int, "float": float, "str": str,
    "list": list, "tuple": tuple, "dict": dict, "set": set, "bool": bool,
    "True": True, "False": False, "None": None,
}

_MAX_OUTPUT_BYTES = 10 * 1024


def _emit(obj):
    _REAL_STDOUT.write(json.dumps(obj))
    _REAL_STDOUT.flush()


def _short(e):
    try:
        return (type(e).__name__ + ": " + str(e))[:200]
    except Exception:
        return type(e).__name__


try:
    payload = json.loads(sys.stdin.read())
    source = payload["source"]
    input_value = payload["input"]
except Exception as _e:
    _emit({"status": "error", "error": ("payload_error: " + _short(_e))[:200]})
    sys.exit(0)

_namespace = {"__builtins__": _ALLOWED_BUILTINS}

try:
    exec(compile(source, "<sandbox>", "exec"), _namespace)
except BaseException as _e:
    _emit({"status": "error", "error": _short(_e)})
    sys.exit(0)

_func = None
for _name, _val in _namespace.items():
    if _name == "__builtins__":
        continue
    if callable(_val):
        _func = _val
        break

if _func is None:
    _emit({"status": "error", "error": "no callable defined"})
    sys.exit(0)

# Swallow any stdout/stderr the function emits so prints can't interleave
# with the JSON envelope on the real stdout pipe.
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

try:
    _result = _func(input_value)
except BaseException as _e:
    _emit({"status": "error", "error": _short(_e)})
    sys.exit(0)

try:
    _serialized = json.dumps(_result)
except BaseException as _e:
    _emit({"status": "error", "error": ("non_json_output: " + _short(_e))[:200]})
    sys.exit(0)

if len(_serialized.encode("utf-8")) > _MAX_OUTPUT_BYTES:
    _emit({"status": "error", "error": "output_too_large"})
    sys.exit(0)

_emit({"status": "ok", "output": _result})
sys.exit(0)
"""


def _make_preexec(memory_mb: int):
    mem_bytes = memory_mb * 1024 * 1024

    def _preexec():
        resource.setrlimit(
            resource.RLIMIT_CPU, (_CPU_SECONDS_HARD_CAP, _CPU_SECONDS_HARD_CAP)
        )
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_NOFILE, (_FD_LIMIT, _FD_LIMIT))
        resource.setrlimit(resource.RLIMIT_NPROC, (_NPROC_LIMIT, _NPROC_LIMIT))
        resource.setrlimit(resource.RLIMIT_FSIZE, (_FSIZE_LIMIT, _FSIZE_LIMIT))

    return _preexec


def _validate_source(source_code: str) -> tuple[bool, str | None]:
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        return False, f"syntax_error: {e.msg} (line {e.lineno})"

    func_defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(func_defs) == 0:
        return False, "no top-level function definition"
    if len(func_defs) > 1:
        return False, f"expected exactly one function, found {len(func_defs)}"
    if len(tree.body) != 1:
        return False, "source contains top-level statements other than a single FunctionDef"
    return True, None


def run_in_sandbox(
    source_code: str,
    input_value: Any,
    timeout_ms: int = 500,
    memory_mb: int = 100,
) -> dict:
    """Execute a pure Python function in an isolated subprocess with resource limits.

    Args:
        source_code: A string containing a single Python function definition.
            The function must take one argument and return one JSON-serializable
            value. The parent validates that the source parses and contains
            exactly one top-level ``FunctionDef`` (and no other top-level
            statements) before spawning the child.
        input_value: The argument to pass to the function. Must be
            JSON-serializable.
        timeout_ms: Wall-clock timeout in milliseconds. **Default 500.** The
            original Day-1 specification used 50ms, which is appropriate for
            native Linux where Python subprocess startup is single-digit ms.
            On WSL2 (Ubuntu on Windows), ``python3 -I -S -c '...'`` startup
            alone is 30-50ms, so a 50ms default guarantees false-positive
            timeouts on correct code. 500ms leaves ~440ms of real execution
            budget after startup and child-script overhead — generous for
            pure functions, tight enough to catch pathological ones quickly.
            Tests that specifically verify fast-kill behavior (e.g.,
            infinite-loop adversarial tests) may pass a smaller value.
        memory_mb: Memory limit in megabytes, enforced via RLIMIT_AS in the
            child's preexec. Default 100.

    Returns:
        A dict with the following keys:

        * ``status``: one of ``"ok"``, ``"timeout"``, ``"error"``,
          ``"crash"``, ``"bad_input"``, ``"bad_source"``.
        * ``output``: the function's return value if ``status == "ok"``,
          else ``None``.
        * ``error``: a short human-readable string describing the error if
          ``status != "ok"``, else ``None``.
        * ``runtime_ms``: wall-clock milliseconds the subprocess ran (int).
          Zero for ``bad_source`` / ``bad_input`` (no subprocess spawned).

    Unexpected errors in the *parent* process (not the child) are not
    silenced; they propagate as exceptions. The status codes are reserved
    for things the sandboxed code did or tried to do.
    """
    ok, err = _validate_source(source_code)
    if not ok:
        return {
            "status": "bad_source",
            "output": None,
            "error": err,
            "runtime_ms": 0,
        }

    try:
        json.dumps(input_value)
    except (TypeError, ValueError) as e:
        return {
            "status": "bad_input",
            "output": None,
            "error": f"input not JSON-serializable: {e}"[:200],
            "runtime_ms": 0,
        }

    payload_bytes = json.dumps(
        {"source": source_code, "input": input_value}
    ).encode("utf-8")

    preexec = _make_preexec(memory_mb)
    timeout_s = timeout_ms / 1000.0

    start = time.monotonic()

    process = subprocess.Popen(
        [_PYTHON3, "-I", "-S", "-c", _CHILD_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=preexec,
        close_fds=True,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
    )

    timed_out = False
    try:
        stdout_bytes, stderr_bytes = process.communicate(
            input=payload_bytes, timeout=timeout_s
        )
    except subprocess.TimeoutExpired:
        timed_out = True
        # Order: kill() (SIGKILL) first, then terminate() (SIGTERM) as fallback.
        # On a healthy Linux kernel SIGKILL is uncatchable and the process MUST
        # die within a handful of milliseconds, so the terminate() branch below
        # is effectively dead code. It is retained as defense-in-depth for
        # exotic failure modes (stuck in uninterruptible D-state on a broken
        # filesystem, ptrace-attached by a debugger, etc.) where SIGKILL can
        # be delayed — in those cases SIGTERM will not help either, but at
        # least we have tried every signal the stdlib exposes before giving up.
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=_POST_KILL_WAIT_S)
        except subprocess.TimeoutExpired:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=_POST_KILL_WAIT_S)
            except subprocess.TimeoutExpired:
                pass
        # Drain pipes so file descriptors close cleanly; the bytes are
        # discarded because on timeout we return a synthetic envelope.
        try:
            stdout_bytes, stderr_bytes = process.communicate(timeout=_POST_KILL_WAIT_S)
        except (subprocess.TimeoutExpired, ValueError):
            stdout_bytes = b""
            stderr_bytes = b""

    elapsed_ms = int((time.monotonic() - start) * 1000)

    if timed_out:
        return {
            "status": "timeout",
            "output": None,
            "error": f"wall-clock timeout after {timeout_ms}ms",
            "runtime_ms": elapsed_ms,
        }

    rc = process.returncode
    stdout_text = stdout_bytes.decode("utf-8", errors="replace")
    stderr_text = stderr_bytes.decode("utf-8", errors="replace")

    try:
        child_response = json.loads(stdout_text) if stdout_text else None
    except (json.JSONDecodeError, ValueError):
        child_response = None

    if not isinstance(child_response, dict) or "status" not in child_response:
        # Child produced no (or unparseable) JSON — most likely killed by
        # RLIMIT_AS / RLIMIT_CPU or segfaulted.
        detail = stderr_text.strip().splitlines()[-1] if stderr_text.strip() else ""
        msg = f"child exited rc={rc}"
        if detail:
            msg = f"{msg}: {detail[:180]}"
        return {
            "status": "crash",
            "output": None,
            "error": msg,
            "runtime_ms": elapsed_ms,
        }

    child_status = child_response.get("status")
    if child_status == "ok":
        return {
            "status": "ok",
            "output": child_response.get("output"),
            "error": None,
            "runtime_ms": elapsed_ms,
        }
    if child_status == "error":
        return {
            "status": "error",
            "output": None,
            "error": str(child_response.get("error", "unknown error"))[:200],
            "runtime_ms": elapsed_ms,
        }

    return {
        "status": "error",
        "output": None,
        "error": f"malformed child response: status={child_status!r}",
        "runtime_ms": elapsed_ms,
    }
