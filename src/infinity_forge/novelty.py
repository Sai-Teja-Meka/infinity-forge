"""Layer 6: behavioral fingerprinting and novelty detection.

An atom's behavioral fingerprint is a 20-element list of strings produced
by running the atom against the 20 frozen probes for its input type. Each
position encodes what happened on that probe:

  * the JSON-canonical form of the atom's output, if the output's shallow
    type matched EXPECTED_OUTPUT_TYPES for the signature's output slot;
  * ``f"__raises__:{ExceptionClassName}"`` if the atom raised (including
    sandbox-level timeouts and crashes, which are encoded as
    ``__raises__:TimeoutError`` and ``__raises__:SandboxCrash``
    respectively — these are behavioral signals, not rejection causes
    at the probe level);
  * ``"__bad_output_type__"`` if the atom returned a value whose shallow
    type doesn't match what the signature promises.

Type-mismatch and duplicate-detection are separate rejection paths. An
atom with any ``__bad_output_type__`` position is rejected outright as
"wrong output type"; it does not compete on novelty. An atom that passes
type checks and matches an existing fingerprint for the same signature
is rejected as "duplicate". Fingerprints from different signatures are
never compared — each signature has its own namespace.

The fingerprint index is a JSONL file. Each line stores one entry:
``{"key": <sig|sig|fingerprint-json>, "source": <str>, "iteration": <int>}``.
It loads into a dict on startup and is appended to per acceptance.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from infinity_forge import sandbox
from infinity_forge.probes import EXPECTED_OUTPUT_TYPES, PROBES

Fingerprint = list[str]

# Per-probe sandbox timeout for fingerprinting. Lower than the sandbox
# default of 500ms so fingerprinting an infinite-loop atom caps at
# ~20 * 200ms = 4s wall-clock rather than 10s. Pure-function probes on
# WSL2 complete in ~30-80ms end-to-end, so 200ms leaves ample headroom
# for non-pathological code while bounding the worst case.
PROBE_TIMEOUT_MS = 200


def _canonical_json(value: Any) -> str:
    """Canonical JSON form: sorted keys, compact separators, deterministic."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _exception_class_from_error(error: str | None) -> str:
    """Extract the exception class name from the sandbox's error string.

    Sandbox error strings have the form ``"ExceptionClass: message"``.
    If parsing fails, return ``"Exception"`` as a conservative fallback.
    """
    if not error:
        return "Exception"
    head = error.split(":", 1)[0].strip()
    if head and head.replace("_", "").isalnum():
        return head
    return "Exception"


def _fingerprint_slot(
    result: dict,
    expected_type: type,
) -> str:
    """Convert one sandbox result into one fingerprint-slot string."""
    status = result["status"]
    if status == "ok":
        output = result["output"]
        if not isinstance(output, expected_type) or (
            expected_type is int and isinstance(output, bool)
        ):
            # bool is an int subclass — rule it out for int slots explicitly.
            return "__bad_output_type__"
        # Mirror of the above: if expected is bool, an int that happens to be
        # 0 or 1 is still not a bool value. isinstance(0, bool) is False, so
        # the default path handles this, but be explicit for readability.
        return _canonical_json(output)
    if status == "timeout":
        return "__raises__:TimeoutError"
    if status == "error":
        return f"__raises__:{_exception_class_from_error(result.get('error'))}"
    # "crash", "validation_error", anything else — encode the status.
    return f"__raises__:Sandbox{status.title().replace('_', '')}"


def compute_fingerprint(source: str, signature: tuple[str, str]) -> Fingerprint:
    """Run ``source`` against the 20 frozen probes for ``signature[0]``.

    Returns a 20-element list of strings per the Layer 6 encoding. Each
    probe spawns its own sandbox subprocess — expect ~20 * 50ms = 1s per
    atom on WSL2. Day 5 can optimize if the generator gets faster than
    fingerprinting.
    """
    input_type, output_type = signature
    if input_type not in PROBES:
        raise ValueError(f"unknown input_type: {input_type!r}")
    if output_type not in EXPECTED_OUTPUT_TYPES:
        raise ValueError(f"unknown output_type: {output_type!r}")

    expected = EXPECTED_OUTPUT_TYPES[output_type]
    probes = PROBES[input_type]

    fingerprint: Fingerprint = []
    for probe in probes:
        result = sandbox.run_in_sandbox(source, probe, timeout_ms=PROBE_TIMEOUT_MS)
        fingerprint.append(_fingerprint_slot(result, expected))
    return fingerprint


def index_key(signature: tuple[str, str], fingerprint: Fingerprint) -> str:
    """Build the composite index key. Signature is part of the key so
    fingerprints from different signatures never collide."""
    return f"{signature[0]}|{signature[1]}|{_canonical_json(fingerprint)}"


def is_novel(
    fingerprint: Fingerprint,
    index: dict[str, tuple[str, int]],
    signature: tuple[str, str],
) -> tuple[bool, str | None]:
    """Return (True, None) if novel; (False, existing_source) if duplicate.

    Duplicate means: same signature AND same fingerprint as an entry already
    in ``index``. Signature is part of the key — an int→int fingerprint and
    an int→bool fingerprint with identical slot values are NOT duplicates.
    """
    key = index_key(signature, fingerprint)
    if key in index:
        existing_source, _iteration = index[key]
        return False, existing_source
    return True, None


def load_index(path: Path) -> dict[str, tuple[str, int]]:
    """Load the fingerprint index from a JSONL side file.

    Empty or missing file returns ``{}``. Malformed lines are skipped
    silently — the file is append-only, so the only way a line is malformed
    is a partial-write crash, and re-fingerprinting the lost atom on the
    next run is cheap.
    """
    index: dict[str, tuple[str, int]] = {}
    if not path.exists():
        return index
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                index[entry["key"]] = (entry["source"], int(entry["iteration"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return index


def append_to_index(
    path: Path,
    key: str,
    source: str,
    iteration: int,
) -> None:
    """Append a single fingerprint entry to the JSONL side file. Flushed."""
    entry = {"key": key, "source": source, "iteration": iteration}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")
        f.flush()
