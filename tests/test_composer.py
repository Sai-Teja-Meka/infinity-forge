"""Gate 4 tests: composer primitives and type-compatible pair enumeration."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from infinity_forge import cascade, sandbox
from infinity_forge.composer import (
    _find_l3_pairs,
    compose_run,
    compose_run_l3,
    compose_source,
    extract_body,
    extract_body_statements,
    extract_param_name,
    find_composable_pairs,
    is_composition_ready,
)


DIVISORS_SRC = "def f(n):\n    return [i for i in range(1, abs(n) + 1) if n % i == 0]"
SUM_SRC = "def f(lst):\n    return sum(lst)"
ABS_STR_SRC = "def f(n):\n    return str(abs(n))"

MULTI_STMT_SRC = (
    "def f(lst):\n"
    "    total = 0\n"
    "    for x in lst:\n"
    "        total = total + x\n"
    "    return total\n"
)


# (a) extract_param_name ---------------------------------------------------


def test_extract_param_name_single_letter():
    assert extract_param_name(DIVISORS_SRC) == "n"


def test_extract_param_name_word():
    assert extract_param_name(SUM_SRC) == "lst"


def test_extract_param_name_multi_statement():
    assert extract_param_name(MULTI_STMT_SRC) == "lst"


def test_extract_param_name_rejects_multi_arg():
    with pytest.raises(ValueError):
        extract_param_name("def f(a, b):\n    return a + b")


# (b) extract_body ---------------------------------------------------------


def test_extract_body_single_expression_is_clean_expr():
    body = extract_body(DIVISORS_SRC)
    assert "return" not in body
    assert "\n" not in body
    # Must be parseable as a stand-alone expression (i.e. valid lambda body).
    ast.parse(body, mode="eval")


def test_extract_body_sum_is_simple_expr():
    assert extract_body(SUM_SRC) == "sum(lst)"


def test_extract_body_multi_statement_preserves_statements():
    body = extract_body(MULTI_STMT_SRC)
    assert "\n" in body
    # Individual statements must parse cleanly.
    ast.parse(body)


# (c) compose_source produces valid Python --------------------------------


def test_compose_source_parses():
    composed = compose_source(DIVISORS_SRC, "n", SUM_SRC, "lst")
    ast.parse(composed)


def test_compose_source_has_expected_lambda_shape():
    composed = compose_source(DIVISORS_SRC, "n", SUM_SRC, "lst")
    assert "def f(n):" in composed
    assert "_a = lambda n:" in composed
    assert "_b = lambda lst:" in composed
    assert "_b(_a(n))" in composed


def test_compose_source_multi_statement_uses_nested_def():
    composed = compose_source(MULTI_STMT_SRC, "lst", ABS_STR_SRC, "n")
    # When either side has a multi-statement body we fall back to nested
    # defs — lambdas can only wrap expressions.
    ast.parse(composed)
    assert "def _a(lst):" in composed
    assert "def _b(n):" in composed


def test_nested_def_path_preserves_return_in_single_expr_atom():
    # DIVISORS is single-expression; MULTI_STMT_SRC forces the nested-def
    # path. The single-expr side must still carry ``return`` inside the
    # nested def, or _a returns None and _b(None) crashes.
    composed = compose_source(DIVISORS_SRC, "n", MULTI_STMT_SRC, "lst")
    ast.parse(composed)
    assert "def _a(n):" in composed
    assert "def _b(lst):" in composed
    # The single-expression atom's nested def must emit ``return``.
    assert "return [i for i in range(1, abs(n) + 1) if n % i == 0]" in composed


def test_nested_def_path_preserves_return_when_b_is_single_expr():
    # Mirror image: A is multi-statement, B is single-expression.
    composed = compose_source(MULTI_STMT_SRC, "lst", ABS_STR_SRC, "n")
    # B's nested def body must include ``return str(abs(n))``.
    assert "return str(abs(n))" in composed


def test_extract_body_statements_preserves_return():
    assert extract_body_statements(SUM_SRC) == "return sum(lst)"
    assert extract_body_statements(DIVISORS_SRC) == (
        "return [i for i in range(1, abs(n) + 1) if n % i == 0]"
    )
    # Multi-statement bodies unchanged by the statement form.
    multi = extract_body_statements(MULTI_STMT_SRC)
    assert "return total" in multi
    assert "total = 0" in multi


def test_nested_def_composition_returns_correct_value():
    # divisors(n) ∘ sum_multi(lst) on 12 must yield 28, not None/crash.
    composed = compose_source(DIVISORS_SRC, "n", MULTI_STMT_SRC, "lst")
    result = sandbox.run_in_sandbox(composed, 12)
    assert result["status"] == "ok", result
    assert result["output"] == 28


def test_nested_def_composition_not_none_false_positive():
    # The false-positive case: _a returns a list, _b consumes a list
    # and happens to tolerate None (via ``or []``). Pre-fix, _a returned
    # None, _b saw None, ``None or []`` yielded [], sum([]) = 0 — a
    # spurious "success" with the wrong answer. Post-fix, the actual
    # list flows through and we get the real sum.
    tolerant_sum = (
        "def f(lst):\n"
        "    items = lst or []\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        total = total + x\n"
        "    return total\n"
    )
    composed = compose_source(DIVISORS_SRC, "n", tolerant_sum, "lst")
    result = sandbox.run_in_sandbox(composed, 12)
    assert result["status"] == "ok", result
    # Correct answer is 28 (1+2+3+4+6+12). A None-driven fallback would
    # give 0 — that's the bug this guards against.
    assert result["output"] == 28, (
        f"expected 28 from real divisors ∘ sum, got {result['output']} — "
        "likely the pre-fix None fallback path"
    )


def test_compose_source_passes_cascade():
    composed = compose_source(DIVISORS_SRC, "n", SUM_SRC, "lst")
    result = cascade.evaluate(composed)
    assert result["accepted"], f"cascade rejected composition: {result}"


# (d) composed function executes correctly --------------------------------


def test_divisors_then_sum_on_12_is_28():
    composed = compose_source(DIVISORS_SRC, "n", SUM_SRC, "lst")
    result = sandbox.run_in_sandbox(composed, 12)
    assert result["status"] == "ok", result
    assert result["output"] == 28  # 1+2+3+4+6+12


def test_divisors_then_sum_on_0_is_0():
    composed = compose_source(DIVISORS_SRC, "n", SUM_SRC, "lst")
    result = sandbox.run_in_sandbox(composed, 0)
    assert result["status"] == "ok"
    assert result["output"] == 0


# (e) find_composable_pairs returns only type-compatible pairs -----------


def test_find_composable_pairs_type_compatibility():
    atoms = [
        {"source": DIVISORS_SRC, "signature": ("int", "list[int]")},
        {"source": SUM_SRC, "signature": ("list[int]", "int")},
        {"source": ABS_STR_SRC, "signature": ("int", "str")},
    ]
    signatures = [
        ("int", "int"),
        ("int", "list[int]"),
        ("list[int]", "int"),
        ("int", "str"),
    ]
    pairs = find_composable_pairs(atoms, signatures)
    # divisors(int→list[int]) ∘ sum(list[int]→int) yields int→int.
    pair_sigs = {composed for _, _, composed in pairs}
    assert ("int", "int") in pair_sigs
    # No mismatched pairs. ABS_STR has output str; nothing consumes str
    # in this fixture, so no pair should have it on the left.
    for a, b, _ in pairs:
        assert tuple(a["signature"])[1] == tuple(b["signature"])[0]


def test_find_composable_pairs_includes_components_correctly():
    atoms = [
        {"source": DIVISORS_SRC, "signature": ("int", "list[int]")},
        {"source": SUM_SRC, "signature": ("list[int]", "int")},
    ]
    signatures = [("int", "int")]
    pairs = find_composable_pairs(atoms, signatures)
    assert len(pairs) == 1
    a, b, composed = pairs[0]
    assert a["source"] == DIVISORS_SRC
    assert b["source"] == SUM_SRC
    assert composed == ("int", "int")


# (f) self-composition excluded --------------------------------------------


def test_find_composable_pairs_excludes_self_composition():
    # An atom of signature (int → int) could compose with itself, but we
    # explicitly exclude that.
    same = {"source": "def f(n):\n    return n + 1", "signature": ("int", "int")}
    pairs = find_composable_pairs([same], [("int", "int")])
    assert pairs == []


def test_find_composable_pairs_allows_different_atoms_same_signature():
    # Two distinct (int → int) atoms can compose in both directions,
    # because their sources differ.
    a = {"source": "def f(n):\n    return n + 1", "signature": ("int", "int")}
    b = {"source": "def f(n):\n    return n * 2", "signature": ("int", "int")}
    pairs = find_composable_pairs([a, b], [("int", "int")])
    # (a, b) and (b, a) both qualify — 2 pairs, no self-composition.
    assert len(pairs) == 2


# (g) composed signature must be in active signatures ---------------------


def test_find_composable_pairs_filters_by_active_signatures():
    atoms = [
        {"source": DIVISORS_SRC, "signature": ("int", "list[int]")},
        {"source": SUM_SRC, "signature": ("list[int]", "int")},
    ]
    # Active list omits ("int", "int"), so the one possible pair is filtered out.
    pairs = find_composable_pairs(atoms, [("int", "list[int]"), ("list[int]", "int")])
    assert pairs == []


# --- compose_run end-to-end (tiny L1 log) ---------------------------------


def _write_l1_log(path: Path, atoms: list[dict]) -> None:
    """Write a minimal L1 log: one accepted record per atom, with fingerprint."""
    from infinity_forge.novelty import compute_fingerprint, index_key, append_to_index

    fp_path = path.with_name(path.stem + ".fingerprints.jsonl")
    with path.open("w", encoding="utf-8") as fh:
        for i, a in enumerate(atoms):
            sig = tuple(a["signature"])
            src = a["source"]
            fp = compute_fingerprint(src, sig)
            rec = {
                "iteration": i,
                "signature": list(sig),
                "extracted_source": src,
                "input_value": None,
                "gate_result": {
                    "accepted": True,
                    "stage": "completed",
                    "reason": None,
                    "value": None,
                    "duration_ms": 0.0,
                    "metadata": {},
                },
                "fingerprint": fp,
            }
            fh.write(json.dumps(rec) + "\n")
            append_to_index(fp_path, index_key(sig, fp), src, i)


def test_compose_run_writes_at_least_one_l2_atom(tmp_path: Path):
    l1 = tmp_path / "l1.jsonl"
    l2 = tmp_path / "l1.l2.jsonl"

    atoms = [
        {"source": DIVISORS_SRC, "signature": ("int", "list[int]")},
        {"source": SUM_SRC, "signature": ("list[int]", "int")},
        {"source": "def f(lst):\n    return len(lst)", "signature": ("list[int]", "int")},
    ]
    _write_l1_log(l1, atoms)

    compose_run(l1, l2, [("int", "int"), ("int", "list[int]"), ("list[int]", "int")])

    assert l2.exists()
    lines = [json.loads(x) for x in l2.read_text().splitlines() if x.strip()]
    assert len(lines) >= 1
    first = lines[0]
    assert first["level"] == 2
    assert "components" in first
    assert len(first["components"]) == 2
    assert first["components"][0]["source"] == DIVISORS_SRC
    # The composed signature must be one of the active signatures we passed.
    assert tuple(first["signature"]) == ("int", "int")


def test_is_composition_ready_clean_fingerprint():
    atom = {
        "source": SUM_SRC,
        "signature": ("list[int]", "int"),
        "fingerprint": ["0", "1", "2", "3", "4", "5"],
    }
    assert is_composition_ready(atom) is True


def test_is_composition_ready_rejects_raises_slot():
    atom = {
        "source": SUM_SRC,
        "signature": ("list[int]", "int"),
        "fingerprint": ["0", "1", "__raises__:ZeroDivisionError", "3"],
    }
    assert is_composition_ready(atom) is False


def test_is_composition_ready_rejects_bad_output_type():
    atom = {
        "source": SUM_SRC,
        "signature": ("list[int]", "int"),
        "fingerprint": ["0", "1", "__bad_output_type__", "3"],
    }
    assert is_composition_ready(atom) is False


def test_is_composition_ready_missing_fingerprint():
    atom = {"source": SUM_SRC, "signature": ("list[int]", "int")}
    assert is_composition_ready(atom) is False
    atom_empty = {
        "source": SUM_SRC,
        "signature": ("list[int]", "int"),
        "fingerprint": [],
    }
    assert is_composition_ready(atom_empty) is False


def test_compose_run_filters_fragile_atoms(tmp_path: Path):
    """A fragile atom (with __raises__) should not participate in composition.

    Concretely: divisors-then-sum normally yields a valid L2 atom. If we
    replace DIVISORS_SRC with a version that raises on one of its probes,
    the composition-readiness filter should exclude it from pair enumeration,
    so no L2 atom is written.
    """
    from infinity_forge.novelty import compute_fingerprint, index_key, append_to_index

    l1 = tmp_path / "l1.jsonl"
    l2 = tmp_path / "l1.l2.jsonl"
    fp_path = l1.with_name(l1.stem + ".fingerprints.jsonl")

    divisors_fp = compute_fingerprint(DIVISORS_SRC, ("int", "list[int]"))
    sum_fp = compute_fingerprint(SUM_SRC, ("list[int]", "int"))
    # Inject a __raises__ slot into the divisors fingerprint, simulating an
    # atom that crashed on one of its 20 probes.
    fragile_fp = list(divisors_fp)
    fragile_fp[0] = "__raises__:ZeroDivisionError"

    records = [
        {
            "iteration": 0,
            "signature": ["int", "list[int]"],
            "extracted_source": DIVISORS_SRC,
            "input_value": None,
            "gate_result": {
                "accepted": True,
                "stage": "completed",
                "reason": None,
                "value": None,
                "duration_ms": 0.0,
                "metadata": {},
            },
            "fingerprint": fragile_fp,
        },
        {
            "iteration": 1,
            "signature": ["list[int]", "int"],
            "extracted_source": SUM_SRC,
            "input_value": None,
            "gate_result": {
                "accepted": True,
                "stage": "completed",
                "reason": None,
                "value": None,
                "duration_ms": 0.0,
                "metadata": {},
            },
            "fingerprint": sum_fp,
        },
    ]
    with l1.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    append_to_index(
        fp_path,
        index_key(("int", "list[int]"), fragile_fp),
        DIVISORS_SRC,
        0,
    )
    append_to_index(fp_path, index_key(("list[int]", "int"), sum_fp), SUM_SRC, 1)

    compose_run(
        l1,
        l2,
        [("int", "int"), ("int", "list[int]"), ("list[int]", "int")],
    )

    if l2.exists():
        lines = [json.loads(x) for x in l2.read_text().splitlines() if x.strip()]
        assert lines == [], (
            f"fragile atom should have been filtered from composition; got {lines}"
        )


def test_compose_run_logs_filtered_count(tmp_path: Path, capsys):
    """The filter-count log line reports how many L1 atoms were excluded."""
    from infinity_forge.novelty import compute_fingerprint, index_key, append_to_index

    l1 = tmp_path / "l1.jsonl"
    l2 = tmp_path / "l1.l2.jsonl"
    fp_path = l1.with_name(l1.stem + ".fingerprints.jsonl")

    divisors_fp = compute_fingerprint(DIVISORS_SRC, ("int", "list[int]"))
    sum_fp = compute_fingerprint(SUM_SRC, ("list[int]", "int"))
    fragile_fp = list(divisors_fp)
    fragile_fp[0] = "__raises__:ZeroDivisionError"

    records = [
        {
            "iteration": 0,
            "signature": ["int", "list[int]"],
            "extracted_source": DIVISORS_SRC,
            "input_value": None,
            "gate_result": {
                "accepted": True,
                "stage": "completed",
                "reason": None,
                "value": None,
                "duration_ms": 0.0,
                "metadata": {},
            },
            "fingerprint": fragile_fp,
        },
        {
            "iteration": 1,
            "signature": ["list[int]", "int"],
            "extracted_source": SUM_SRC,
            "input_value": None,
            "gate_result": {
                "accepted": True,
                "stage": "completed",
                "reason": None,
                "value": None,
                "duration_ms": 0.0,
                "metadata": {},
            },
            "fingerprint": sum_fp,
        },
    ]
    with l1.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    append_to_index(
        fp_path, index_key(("int", "list[int]"), fragile_fp), DIVISORS_SRC, 0
    )
    append_to_index(fp_path, index_key(("list[int]", "int"), sum_fp), SUM_SRC, 1)

    compose_run(
        l1,
        l2,
        [("int", "int"), ("int", "list[int]"), ("list[int]", "int")],
    )
    captured = capsys.readouterr()
    # Filter log: 1 of 2 atoms excluded, 1 __raises__, 0 __bad_output_type__.
    assert "filtered 1 of 2 L1 atoms as not composition-ready" in captured.out
    assert "1 with __raises__" in captured.out
    assert "0 with __bad_output_type__" in captured.out


def test_compose_run_writes_failures_file(tmp_path: Path):
    """Rejected compositions are appended to a .failures.jsonl alongside accepts."""
    l1 = tmp_path / "l1.jsonl"
    l2 = tmp_path / "l1.l2.jsonl"

    # Two L1 atoms that compose into a fingerprint-duplicate of a third L1
    # atom — that guarantees at least one rejection (layer_6 duplicate).
    sigma_src = "def f(n):\n    return sum(i for i in range(1, abs(n) + 1) if n % i == 0)"
    atoms = [
        {"source": DIVISORS_SRC, "signature": ("int", "list[int]")},
        {"source": SUM_SRC, "signature": ("list[int]", "int")},
        {"source": sigma_src, "signature": ("int", "int")},
    ]
    _write_l1_log(l1, atoms)

    compose_run(l1, l2, [("int", "int"), ("int", "list[int]"), ("list[int]", "int")])

    failures_path = l2.with_suffix(".failures.jsonl")
    assert failures_path.exists(), "failures file should be written"
    records = [
        json.loads(line) for line in failures_path.read_text().splitlines() if line.strip()
    ]
    assert records, "failures file should contain at least one record"
    required = {
        "atom_a",
        "atom_b",
        "composed_source",
        "composed_signature",
        "rejection_stage",
        "rejection_reason",
    }
    for rec in records:
        assert required <= set(rec.keys()), f"missing keys in {rec}"
        assert "source" in rec["atom_a"] and "signature" in rec["atom_a"]
        assert "source" in rec["atom_b"] and "signature" in rec["atom_b"]
        assert isinstance(rec["composed_signature"], list)
        assert rec["rejection_stage"] in {
            "sandbox",
            "layer_1",
            "layer_2",
            "layer_3",
            "layer_4",
            "layer_5",
            "layer_6",
            "novelty",
            "canonical",
            "extract_fail",
        }
    # At least one record should describe a layer_6 fingerprint-duplicate
    # (the divisors ∘ sum ≡ sigma case).
    assert any(rec["rejection_stage"] == "layer_6" for rec in records)


def test_compose_run_rejects_duplicates_of_level1(tmp_path: Path):
    l1 = tmp_path / "l1.jsonl"
    l2 = tmp_path / "l1.l2.jsonl"

    # Include an int→int atom whose behavior matches `divisors ∘ sum`: the
    # sigma-of-divisors function. On input 12 it yields 28, like the
    # composition. When Layer 6 fingerprints match, the composition must
    # be rejected as duplicate.
    sigma_src = "def f(n):\n    return sum(i for i in range(1, abs(n) + 1) if n % i == 0)"
    atoms = [
        {"source": DIVISORS_SRC, "signature": ("int", "list[int]")},
        {"source": SUM_SRC, "signature": ("list[int]", "int")},
        {"source": sigma_src, "signature": ("int", "int")},
    ]
    _write_l1_log(l1, atoms)

    compose_run(l1, l2, [("int", "int"), ("int", "list[int]"), ("list[int]", "int")])

    # The composition divisors ∘ sum should have been rejected as a
    # fingerprint-duplicate of sigma_src — so no ("int", "int") L2 atom
    # is written.
    if l2.exists():
        lines = [json.loads(x) for x in l2.read_text().splitlines() if x.strip()]
        for rec in lines:
            assert tuple(rec["signature"]) != ("int", "int"), (
                "composition should have been rejected as duplicate of L1 sigma"
            )


# --- Level 3 composition --------------------------------------------------


def _write_l2_log(path: Path, atoms: list[dict]) -> None:
    """Write a minimal L2 log: one accepted record per atom, with fingerprint."""
    from infinity_forge.novelty import (
        append_to_index,
        compute_fingerprint,
        index_key,
    )

    fp_path = path.with_name(path.stem + ".fingerprints.jsonl")
    with path.open("w", encoding="utf-8") as fh:
        for i, a in enumerate(atoms):
            sig = tuple(a["signature"])
            src = a["source"]
            fp = a.get("fingerprint") or compute_fingerprint(src, sig)
            rec = {
                "iteration": i,
                "level": 2,
                "signature": list(sig),
                "extracted_source": src,
                "input_value": None,
                "gate_result": {
                    "accepted": True,
                    "stage": "completed",
                    "reason": None,
                    "value": None,
                    "duration_ms": 0.0,
                    "metadata": {},
                },
                "fingerprint": fp,
                "components": [],
            }
            fh.write(json.dumps(rec) + "\n")
            append_to_index(fp_path, index_key(sig, fp), src, i)


SQUARE_SRC = "def f(n):\n    return n * n"
INCR_SRC = "def f(n):\n    return n + 1"
DOUBLE_SRC = "def f(n):\n    return n * 2"
IDENTITY_SRC = "def f(n):\n    return n"


def test_find_l3_pairs_enumerates_both_directions():
    l1 = [
        {"source": INCR_SRC, "signature": ("int", "int"), "fingerprint": ["x"]},
    ]
    l2 = [
        {"source": SQUARE_SRC, "signature": ("int", "int"), "fingerprint": ["y"]},
    ]
    pairs = _find_l3_pairs(l1, l2, [("int", "int")])
    sources = {(a["source"], b["source"]) for a, b, _ in pairs}
    # Both (square, incr) and (incr, square) must appear.
    assert (SQUARE_SRC, INCR_SRC) in sources, (
        f"expected L2-inner → L1-outer; got {sources}"
    )
    assert (INCR_SRC, SQUARE_SRC) in sources, (
        f"expected L1-inner → L2-outer; got {sources}"
    )
    assert len(pairs) == 2


def test_find_l3_pairs_filters_by_active_signatures():
    l1 = [
        {"source": ABS_STR_SRC, "signature": ("int", "str"), "fingerprint": ["x"]},
    ]
    l2 = [
        {"source": SQUARE_SRC, "signature": ("int", "int"), "fingerprint": ["y"]},
    ]
    # SQUARE(int→int) inner ∘ ABS_STR(int→str) outer would compose to
    # (int, str), but the active list omits that signature, so the pair
    # is filtered out. The reverse direction is type-incompatible
    # (str ≠ int).
    pairs = _find_l3_pairs(l1, l2, [("int", "int")])
    assert pairs == []


def test_find_l3_pairs_excludes_self_composition():
    same = {"source": SQUARE_SRC, "signature": ("int", "int"), "fingerprint": ["x"]}
    pairs = _find_l3_pairs([same], [same], [("int", "int")])
    assert pairs == []


def test_compose_run_l3_writes_three_function_chain(tmp_path: Path):
    l1 = tmp_path / "l1.jsonl"
    l2 = tmp_path / "l1.l2.jsonl"
    l3 = tmp_path / "l1.l3.jsonl"

    l1_atoms = [
        {"source": INCR_SRC, "signature": ("int", "int")},
        {"source": DOUBLE_SRC, "signature": ("int", "int")},
    ]
    _write_l1_log(l1, l1_atoms)
    _write_l2_log(l2, [{"source": SQUARE_SRC, "signature": ("int", "int")}])

    compose_run_l3(l1, l2, l3, [("int", "int")])

    assert l3.exists()
    lines = [json.loads(x) for x in l3.read_text().splitlines() if x.strip()]
    assert lines, "expected at least one L3 atom"
    rec = lines[0]
    assert rec["level"] == 3
    assert "components" in rec
    assert len(rec["components"]) == 2
    # The accepted source must execute correctly via the sandbox: the
    # whole composition is a real three-function chain.
    from infinity_forge import sandbox

    result = sandbox.run_in_sandbox(rec["extracted_source"], 3)
    assert result["status"] == "ok", result


def test_compose_run_l3_components_cover_both_directions(tmp_path: Path):
    """Across all accepted L3 atoms, both (L2, L1) and (L1, L2) appear."""
    l1 = tmp_path / "l1.jsonl"
    l2 = tmp_path / "l1.l2.jsonl"
    l3 = tmp_path / "l1.l3.jsonl"

    l1_atoms = [
        {"source": INCR_SRC, "signature": ("int", "int")},
        {"source": DOUBLE_SRC, "signature": ("int", "int")},
    ]
    _write_l1_log(l1, l1_atoms)
    _write_l2_log(l2, [{"source": SQUARE_SRC, "signature": ("int", "int")}])

    compose_run_l3(l1, l2, l3, [("int", "int")])

    lines = [json.loads(x) for x in l3.read_text().splitlines() if x.strip()]
    saw_l2_inner = False
    saw_l1_inner = False
    for rec in lines:
        first_src = rec["components"][0]["source"]
        if first_src == SQUARE_SRC:
            saw_l2_inner = True
        elif first_src in (INCR_SRC, DOUBLE_SRC):
            saw_l1_inner = True
    assert saw_l2_inner, f"no L3 atom had L2 as inner component; got {lines}"
    assert saw_l1_inner, f"no L3 atom had L1 as inner component; got {lines}"


def test_compose_run_l3_dedupes_against_level1(tmp_path: Path):
    """An L3 candidate whose fingerprint matches an existing L1 atom is rejected."""
    l1 = tmp_path / "l1.jsonl"
    l2 = tmp_path / "l1.l2.jsonl"
    l3 = tmp_path / "l1.l3.jsonl"

    # incr is in L1; identity is the only L2 atom. Both compositions of
    # the pair (identity, incr) and (incr, identity) yield n+1, whose
    # fingerprint matches incr — so every L3 candidate must be rejected
    # as a layer_6 fingerprint duplicate.
    _write_l1_log(l1, [{"source": INCR_SRC, "signature": ("int", "int")}])
    _write_l2_log(l2, [{"source": IDENTITY_SRC, "signature": ("int", "int")}])

    compose_run_l3(l1, l2, l3, [("int", "int")])

    if l3.exists():
        lines = [json.loads(x) for x in l3.read_text().splitlines() if x.strip()]
        assert lines == [], (
            f"L3 candidates duplicating L1 atoms must be rejected; got {lines}"
        )

    failures_path = l3.with_suffix(".failures.jsonl")
    assert failures_path.exists()
    failures = [
        json.loads(x) for x in failures_path.read_text().splitlines() if x.strip()
    ]
    # At least one rejection should be canonical-or-fingerprint duplicate
    # of the L1 incr atom.
    assert any(
        rec["rejection_stage"] in {"layer_6", "canonical"} for rec in failures
    ), f"expected canonical/layer_6 rejection vs L1; got {failures}"


def test_compose_run_l3_dedupes_against_level2(tmp_path: Path):
    """An L3 candidate whose fingerprint matches an existing L2 atom is rejected."""
    l1 = tmp_path / "l1.jsonl"
    l2 = tmp_path / "l1.l2.jsonl"
    l3 = tmp_path / "l1.l3.jsonl"

    # L1: identity. L2: square. The pair (identity, square) and
    # (square, identity) both compute n*n — fingerprint matches the
    # existing L2 square atom, so both L3 candidates must dedupe.
    _write_l1_log(l1, [{"source": IDENTITY_SRC, "signature": ("int", "int")}])
    _write_l2_log(l2, [{"source": SQUARE_SRC, "signature": ("int", "int")}])

    compose_run_l3(l1, l2, l3, [("int", "int")])

    if l3.exists():
        lines = [json.loads(x) for x in l3.read_text().splitlines() if x.strip()]
        assert lines == [], (
            f"L3 candidates duplicating L2 atoms must be rejected; got {lines}"
        )


def test_compose_run_l3_filters_fragile_l2_atoms(tmp_path: Path):
    """An L2 atom with __raises__ in its fingerprint must not enter any L3 pair."""
    from infinity_forge.novelty import compute_fingerprint

    l1 = tmp_path / "l1.jsonl"
    l2 = tmp_path / "l1.l2.jsonl"
    l3 = tmp_path / "l1.l3.jsonl"

    _write_l1_log(l1, [{"source": INCR_SRC, "signature": ("int", "int")}])

    # Inject __raises__ into the L2 atom's fingerprint. This is the only
    # L2 atom, so once it's filtered out there are no L3 pairs at all.
    square_fp = list(compute_fingerprint(SQUARE_SRC, ("int", "int")))
    square_fp[0] = "__raises__:ValueError"
    _write_l2_log(
        l2,
        [
            {
                "source": SQUARE_SRC,
                "signature": ("int", "int"),
                "fingerprint": square_fp,
            }
        ],
    )

    compose_run_l3(l1, l2, l3, [("int", "int")])

    if l3.exists():
        lines = [json.loads(x) for x in l3.read_text().splitlines() if x.strip()]
        assert lines == [], (
            f"fragile L2 atoms must be filtered from L3 enumeration; got {lines}"
        )


def test_compose_run_l3_logs_filter_lines_for_both_layers(tmp_path: Path, capsys):
    """The filter-count diagnostic appears for both L1 and L2."""
    l1 = tmp_path / "l1.jsonl"
    l2 = tmp_path / "l1.l2.jsonl"
    l3 = tmp_path / "l1.l3.jsonl"

    _write_l1_log(l1, [{"source": INCR_SRC, "signature": ("int", "int")}])
    _write_l2_log(l2, [{"source": SQUARE_SRC, "signature": ("int", "int")}])

    compose_run_l3(l1, l2, l3, [("int", "int")])
    captured = capsys.readouterr()
    assert "[composer-l3] filtered 0 of 1 L1 atoms" in captured.out
    assert "[composer-l3] filtered 0 of 1 L2 atoms" in captured.out
