"""Gate 4 tests: Layer 6 fingerprint computation, novelty checks, index I/O."""
from __future__ import annotations

import json

import pytest

from infinity_forge.novelty import (
    append_to_index,
    compute_fingerprint,
    index_key,
    is_novel,
    load_index,
)
from infinity_forge.probes import PROBES


ABS_V1 = "def f(n):\n    return abs(n)"
ABS_V2 = "def f(n):\n    return n if n >= 0 else -n"

IDENTITY_INT = "def f(n):\n    return n"

RAISES_ON_ZERO = "def f(n):\n    return 1 // n"

RETURNS_STR_FOR_INT_SIG = "def f(n):\n    return str(n)"


# ---------- (a) behavioral equivalence: syntactically different, same behavior ----------

def test_behaviorally_equivalent_functions_have_same_fingerprint():
    fp1 = compute_fingerprint(ABS_V1, ("int", "int"))
    fp2 = compute_fingerprint(ABS_V2, ("int", "int"))
    assert fp1 == fp2


def test_fingerprint_is_twenty_elements():
    fp = compute_fingerprint(IDENTITY_INT, ("int", "int"))
    assert len(fp) == 20


def test_distinct_behaviors_have_different_fingerprints():
    fp_abs = compute_fingerprint(ABS_V1, ("int", "int"))
    fp_id = compute_fingerprint(IDENTITY_INT, ("int", "int"))
    assert fp_abs != fp_id


# ---------- (b) raises encoding ----------

def test_function_that_raises_on_probe_zero_has_raises_at_position_zero():
    # PROBES['int'][0] == 0, so 1 // 0 raises ZeroDivisionError on that probe.
    assert PROBES["int"][0] == 0
    fp = compute_fingerprint(RAISES_ON_ZERO, ("int", "int"))
    assert fp[0].startswith("__raises__:")
    assert "ZeroDivisionError" in fp[0]


def test_timeout_encoded_as_raises_timeout_error():
    # An infinite loop should timeout; sandbox default 500ms is the limit.
    src = "def f(n):\n    while True:\n        pass"
    fp = compute_fingerprint(src, ("int", "int"))
    # Every probe times out → every slot is __raises__:TimeoutError.
    for slot in fp:
        assert slot == "__raises__:TimeoutError"


# ---------- (c) bad output type encoding ----------

def test_wrong_output_type_encoded_as_bad_output_type():
    # Signature claims int→int but function returns str.
    fp = compute_fingerprint(RETURNS_STR_FOR_INT_SIG, ("int", "int"))
    # All 20 probes return str on a signature promising int.
    for slot in fp:
        assert slot == "__bad_output_type__"


def test_bool_is_not_treated_as_int_for_int_output_slot():
    # bool is an int subclass in Python, but a bool-returning function on an
    # int-signature should NOT pass the output-type check.
    src = "def f(n):\n    return n > 0"
    fp = compute_fingerprint(src, ("int", "int"))
    for slot in fp:
        assert slot == "__bad_output_type__"


# ---------- (d) duplicate detection ----------

def test_is_novel_returns_false_for_same_fingerprint_same_signature():
    fp = compute_fingerprint(ABS_V1, ("int", "int"))
    key = index_key(("int", "int"), fp)
    index = {key: (ABS_V1, 7)}
    novel, existing = is_novel(fp, index, ("int", "int"))
    assert novel is False
    assert existing == ABS_V1


def test_is_novel_returns_true_for_new_fingerprint():
    fp_abs = compute_fingerprint(ABS_V1, ("int", "int"))
    fp_id = compute_fingerprint(IDENTITY_INT, ("int", "int"))
    key_abs = index_key(("int", "int"), fp_abs)
    index = {key_abs: (ABS_V1, 3)}
    novel, existing = is_novel(fp_id, index, ("int", "int"))
    assert novel is True
    assert existing is None


# ---------- (e) cross-signature namespace isolation ----------

def test_is_novel_treats_same_fingerprint_different_signature_as_novel():
    # Put a fingerprint under ("int","int"); same fingerprint checked against
    # ("int","bool") must be considered novel because the key differs.
    fp = compute_fingerprint(IDENTITY_INT, ("int", "int"))
    key_int_int = index_key(("int", "int"), fp)
    index = {key_int_int: (IDENTITY_INT, 1)}
    novel, existing = is_novel(fp, index, ("int", "bool"))
    assert novel is True
    assert existing is None


# ---------- (f) index round-trip through a tmp file ----------

def test_load_index_missing_file_returns_empty(tmp_path):
    missing = tmp_path / "does_not_exist.jsonl"
    assert load_index(missing) == {}


def test_load_index_empty_file_returns_empty(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert load_index(empty) == {}


def test_append_and_load_index_round_trip(tmp_path):
    path = tmp_path / "fp_index.jsonl"
    append_to_index(path, "int|int|[\"0\"]", "def f(n):\n    return 0", 0)
    append_to_index(path, "int|int|[\"1\"]", "def f(n):\n    return 1", 1)
    append_to_index(path, "dict|int|[\"x\"]", "def f(d):\n    return len(d)", 2)

    index = load_index(path)
    assert len(index) == 3
    assert index["int|int|[\"0\"]"] == ("def f(n):\n    return 0", 0)
    assert index["int|int|[\"1\"]"] == ("def f(n):\n    return 1", 1)
    assert index["dict|int|[\"x\"]"] == ("def f(d):\n    return len(d)", 2)


def test_index_key_is_deterministic():
    fp = ["1", "2", "3"]
    k1 = index_key(("int", "int"), fp)
    k2 = index_key(("int", "int"), fp)
    assert k1 == k2


def test_index_key_differs_by_signature():
    fp = ["1", "2", "3"]
    k_int_int = index_key(("int", "int"), fp)
    k_int_bool = index_key(("int", "bool"), fp)
    assert k_int_int != k_int_bool


def test_load_index_skips_malformed_lines(tmp_path):
    path = tmp_path / "mixed.jsonl"
    entry = {"key": "int|int|[]", "source": "def f(n):\n    return n", "iteration": 0}
    path.write_text(
        "not valid json\n"
        + json.dumps(entry) + "\n"
        + "{\"missing\": \"fields\"}\n",
        encoding="utf-8",
    )
    index = load_index(path)
    assert len(index) == 1
    assert "int|int|[]" in index
