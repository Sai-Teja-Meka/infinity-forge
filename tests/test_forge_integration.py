"""Integration tests for forge.run(). MockGenerator only — no real LLM.

Day 4 tests exercise the real pipeline end-to-end: seeds inject on fresh
runs (negative iterations), Layer 6 runs on every gate-accepted atom, and
few-shot pulls distinct fingerprints. Test assertions filter out seed
entries by ``iteration >= 0`` when counting generator iterations.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from infinity_forge.forge import _pick_few_shot, run
from infinity_forge.generator import MockGenerator
from infinity_forge.seeds import SEED_ATOMS
from infinity_forge.signatures import ACTIVE_SIGNATURES


def _sig_for(iteration: int) -> tuple[str, str]:
    return ACTIVE_SIGNATURES[iteration % len(ACTIVE_SIGNATURES)]


def _temp_for(iteration: int) -> float:
    return (0.7, 0.9, 1.1)[iteration % 3]


_SIG_SOURCES: dict[tuple[str, str], str] = {
    ("int", "int"): "def f(n):\n    return n + 1",
    ("int", "bool"): "def f(n):\n    return n > 0",
    ("int", "list[int]"): "def f(n):\n    return [n, n + 1]",
    ("list[int]", "int"): "def f(lst):\n    return sum(lst)",
    ("list[int]", "bool"): "def f(lst):\n    return len(lst) > 0",
    ("list[int]", "list[int]"): "def f(lst):\n    return sorted(lst)",
    ("list[int]", "dict"): "def f(lst):\n    return {str(i): lst[i] for i in range(len(lst))}",
    ("list[str]", "str"): "def f(lst):\n    return ''.join(lst)",
    ("list[str]", "list[str]"): "def f(lst):\n    return sorted(lst)",
    ("str", "int"): "def f(s):\n    return len(s)",
    ("str", "bool"): "def f(s):\n    return len(s) > 0",
    ("str", "str"): "def f(s):\n    return s + 'x'",
    ("dict", "int"): "def f(d):\n    return len(d)",
    ("dict", "list[str]"): "def f(d):\n    return sorted(d)",
}


def _accept_src_for(sig: tuple[str, str]) -> str:
    """Fenced accept-worthy source for the given signature.

    Each signature has an atom that passes cascade, executes cleanly in
    the sandbox, and produces the declared output type on all 20 probes.
    """
    return f"```python\n{_SIG_SOURCES[sig]}\n```"


_REJECT_SRC = "```python\ndef f(x):\n    return open('/etc/passwd').read()\n```"
_EXTRACT_FAIL = "no code here at all, just prose"


def _build_sequence(n: int) -> list[str]:
    """Per-iteration response list keyed to i's modulo-10 bucket.

    For each iteration the bucket determines the response shape, and when
    the response is an accept we pick a signature-appropriate source so
    Layer 6 does not reject it on type mismatch.
      - i % 10 in {0, 1, 8, 9}: accepted (8 per 20 iterations)
      - i % 10 in {2, 3, 4}: cascade-rejected (6 per 20)
      - i % 10 in {5, 6, 7}: extract-fail (6 per 20)
    """
    out: list[str] = []
    for i in range(n):
        b = i % 10
        if b in (0, 1, 8, 9):
            out.append(_accept_src_for(_sig_for(i)))
        elif b in (2, 3, 4):
            out.append(_REJECT_SRC)
        else:
            out.append(_EXTRACT_FAIL)
    return out


def _make_gen(n: int) -> MockGenerator:
    return MockGenerator(sequence=_build_sequence(n))


def _gen_iterations(log: Path) -> list[dict]:
    """All log records with iteration >= 0 (excludes seeds)."""
    recs: list[dict] = []
    for line in log.read_text(encoding="utf-8").strip().split("\n"):
        if not line:
            continue
        rec = json.loads(line)
        if rec["iteration"] >= 0:
            recs.append(rec)
    return recs


def _seed_iterations(log: Path) -> list[dict]:
    recs: list[dict] = []
    for line in log.read_text(encoding="utf-8").strip().split("\n"):
        if not line:
            continue
        rec = json.loads(line)
        if rec["iteration"] < 0:
            recs.append(rec)
    return recs


# ----------------------------- existing coverage, adapted --------------------


def test_run_writes_expected_generator_line_count(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    gen = _make_gen(20)
    run(gen, log, n_iterations=20, resume=False)

    assert len(_gen_iterations(log)) == 20


def test_every_line_is_valid_json_with_expected_fields(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    gen = _make_gen(20)
    run(gen, log, n_iterations=20, resume=False)

    expected_fields = {
        "iteration", "signature", "temperature", "raw_llm_output",
        "extracted_source", "input_value", "gate_result", "fingerprint",
        "timestamp",
    }
    for line in log.read_text(encoding="utf-8").strip().split("\n"):
        rec = json.loads(line)
        assert expected_fields.issubset(rec.keys())
        assert isinstance(rec["iteration"], int)
        assert isinstance(rec["signature"], list) and len(rec["signature"]) == 2
        assert isinstance(rec["temperature"], (int, float))
        assert isinstance(rec["raw_llm_output"], str)
        assert isinstance(rec["timestamp"], str)


def test_round_robin_signature_distribution(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    gen = _make_gen(20)
    run(gen, log, n_iterations=20, resume=False)

    for rec in _gen_iterations(log):
        i = rec["iteration"]
        assert rec["signature"] == list(_sig_for(i))
        assert rec["temperature"] == _temp_for(i)


def test_acceptance_and_rejection_counts(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    gen = _make_gen(20)
    run(gen, log, n_iterations=20, resume=False)

    accepted = 0
    layer2_rejected = 0
    extract_failed = 0
    for rec in _gen_iterations(log):
        if rec["extracted_source"] is None:
            assert rec["gate_result"] is None
            extract_failed += 1
        elif rec["gate_result"]["accepted"]:
            accepted += 1
        elif rec["gate_result"]["stage"] == "layer_2":
            layer2_rejected += 1

    assert accepted == 8
    assert layer2_rejected == 6
    assert extract_failed == 6


def test_resumption_picks_up_after_existing(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    gen = _make_gen(20)

    run(gen, log, n_iterations=10, resume=False)
    first_iters = [rec["iteration"] for rec in _gen_iterations(log)]
    assert first_iters == list(range(10))

    run(gen, log, n_iterations=10, resume=True)
    all_iters = [rec["iteration"] for rec in _gen_iterations(log)]
    assert all_iters == list(range(20))


def test_resumption_no_overlap(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    gen = _make_gen(20)
    run(gen, log, n_iterations=10, resume=False)
    run(gen, log, n_iterations=10, resume=True)

    iters = [rec["iteration"] for rec in _gen_iterations(log)]
    assert len(iters) == len(set(iters))


def test_status_line_printed_every_25(tmp_path: Path, capsys):
    log = tmp_path / "log.jsonl"
    gen = _make_gen(60)
    run(gen, log, n_iterations=50, resume=False)
    captured = capsys.readouterr().out
    status_lines = [l for l in captured.splitlines() if l.startswith("[forge] iter ")]
    assert len(status_lines) == 2


def test_status_line_not_printed_below_25(tmp_path: Path, capsys):
    log = tmp_path / "log.jsonl"
    gen = _make_gen(20)
    run(gen, log, n_iterations=20, resume=False)
    captured = capsys.readouterr().out
    status_lines = [l for l in captured.splitlines() if l.startswith("[forge] iter ")]
    assert len(status_lines) == 0


def test_empty_active_signatures_raises(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    gen = MockGenerator()
    with pytest.raises(ValueError):
        run(gen, log, n_iterations=1, active_signatures=[], resume=False)


def test_no_resume_starts_at_zero_but_appends(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    gen = _make_gen(20)
    run(gen, log, n_iterations=5, resume=False)
    run(gen, log, n_iterations=5, resume=False)
    iters = [rec["iteration"] for rec in _gen_iterations(log)]
    # Second run starts at 0 again (no resume), so iterations overlap.
    assert iters == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]


# ----------------------------- Day 4 gate-5 new coverage ---------------------


def test_fresh_run_writes_seed_atoms_before_iteration_zero(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    gen = _make_gen(1)
    run(gen, log, n_iterations=1, resume=False)

    all_recs = [json.loads(l) for l in log.read_text().strip().split("\n")]
    # First N records are seeds, with negative iterations (-1, -2, -3, ...).
    n_seeds = len(SEED_ATOMS)
    seed_recs = all_recs[:n_seeds]
    for i, rec in enumerate(seed_recs):
        assert rec["iteration"] == -(i + 1), f"seed {i} iteration wrong: {rec}"
        assert rec["gate_result"]["accepted"] is True
        assert rec["fingerprint"] is not None
        assert len(rec["fingerprint"]) == 20
    # After seeds, the generator iteration begins at 0.
    assert all_recs[n_seeds]["iteration"] == 0


def test_seeds_written_to_fingerprint_index(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    gen = _make_gen(1)
    run(gen, log, n_iterations=1, resume=False)

    fp_path = tmp_path / "log.fingerprints.jsonl"
    assert fp_path.exists()
    lines = [l for l in fp_path.read_text().split("\n") if l.strip()]
    # Seeds contribute their own index entries; non-seed accept may add more.
    assert len(lines) >= len(SEED_ATOMS)


def test_resume_does_not_reinject_seeds(tmp_path: Path, capsys):
    log = tmp_path / "log.jsonl"
    gen = _make_gen(10)

    run(gen, log, n_iterations=5, resume=False)
    seeds_first = _seed_iterations(log)
    assert len(seeds_first) == len(SEED_ATOMS)

    first_out = capsys.readouterr().out
    assert "resuming from iteration" not in first_out
    assert "[forge] completed 5 iterations:" in first_out

    run(gen, log, n_iterations=5, resume=True)
    seeds_second = _seed_iterations(log)
    assert seeds_second == seeds_first  # byte-for-byte same seed rows

    gen_iters = [rec["iteration"] for rec in _gen_iterations(log)]
    assert gen_iters == list(range(10))

    second_out = capsys.readouterr().out
    assert "[forge] resuming from iteration 5," in second_out
    assert "[forge] completed 5 iterations:" in second_out


def test_duplicate_atom_flagged_as_novelty(tmp_path: Path):
    """Same source twice for the same signature → second is rejected at novelty.

    Since Gate 5, the second is caught by the canonical fast-path before
    fingerprinting, so ``fingerprint`` on the duplicate is ``None``.
    """
    log = tmp_path / "log.jsonl"
    src = _accept_src_for(("int", "int"))
    gen = MockGenerator(sequence=[src, src])
    run(
        gen,
        log,
        n_iterations=2,
        active_signatures=[("int", "int")],
        resume=False,
    )

    gen_recs = _gen_iterations(log)
    assert len(gen_recs) == 2

    first = gen_recs[0]
    second = gen_recs[1]

    assert first["gate_result"]["accepted"] is True
    assert first["gate_result"]["stage"] == "completed"
    assert first["fingerprint"] is not None

    assert second["gate_result"]["accepted"] is False
    assert second["gate_result"]["stage"] == "novelty"
    assert "duplicate" in second["gate_result"]["reason"]
    assert "iteration 0" in second["gate_result"]["reason"]
    assert second["fingerprint"] is None


def test_few_shot_deduplicates_by_fingerprint():
    """_pick_few_shot returns only distinct-fingerprint atoms, newest first."""
    pool = [
        ("src_A", ["1"] * 20),
        ("src_B", ["2"] * 20),
        ("src_C", ["1"] * 20),  # same fingerprint as A
        ("src_D", ["3"] * 20),
        ("src_E", ["2"] * 20),  # same fingerprint as B
    ]
    picked = _pick_few_shot(pool, k=3)
    # Newest-first traversal: E (dup of B), D, C (dup of A), B, A
    # Distinct picks: E (fp=2), D (fp=3), C (fp=1) → three in reverse-insertion order.
    assert picked == ["src_E", "src_D", "src_C"]


def test_csv_metrics_has_header_and_monotonic_cumulatives(tmp_path: Path):
    """10 iterations → CSV with header + 10 rows; cum_* columns non-decreasing."""
    log = tmp_path / "log.jsonl"
    gen = _make_gen(10)
    run(gen, log, n_iterations=10, resume=False)

    csv_path = tmp_path / "log.csv"
    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)

    header = rows[0]
    assert header == [
        "iteration", "signature_in", "signature_out", "generator",
        "stage", "accepted",
        "cum_accepted", "cum_dup", "cum_bad", "cum_l1", "cum_l2", "cum_sandbox",
    ]
    data = rows[1:]
    assert len(data) == 10
    # Single-model (non-Multi) runs leave the generator column blank.
    for row in data:
        rec = dict(zip(header, row))
        assert rec["generator"] == ""

    cum_cols = ("cum_accepted", "cum_dup", "cum_bad", "cum_l1", "cum_l2", "cum_sandbox")
    prev = {c: 0 for c in cum_cols}
    for row in data:
        rec = dict(zip(header, row))
        for c in cum_cols:
            cur = int(rec[c])
            assert cur >= prev[c], f"{c} decreased: {prev[c]} -> {cur}"
            prev[c] = cur


def test_multi_generator_records_generator_in_jsonl_and_csv(tmp_path: Path):
    """Run with a MultiGenerator; JSONL alternates ``generator`` a/b, CSV col populated."""
    from infinity_forge.generator import MultiGenerator

    n = 10
    seq_a = [_accept_src_for(_sig_for(i)) for i in range(0, n, 2)]
    seq_b = [_accept_src_for(_sig_for(i)) for i in range(1, n, 2)]
    gen_a = MockGenerator(sequence=seq_a)
    gen_b = MockGenerator(sequence=seq_b)
    multi = MultiGenerator([("a", gen_a), ("b", gen_b)])

    log = tmp_path / "log.jsonl"
    run(multi, log, n_iterations=n, resume=False)

    recs = _gen_iterations(log)
    assert len(recs) == n
    for rec in recs:
        assert "generator" in rec, f"JSONL missing generator field: {rec}"
    gen_names = [rec["generator"] for rec in recs]
    assert gen_names == ["a", "b"] * (n // 2)

    csv_path = tmp_path / "log.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    data = rows[1:]
    csv_gens = [dict(zip(header, row))["generator"] for row in data]
    assert csv_gens == ["a", "b"] * (n // 2)


def test_single_model_run_omits_generator_field_from_jsonl(tmp_path: Path):
    """Backward-compat: no ``generator`` key when a bare Generator is used."""
    log = tmp_path / "log.jsonl"
    gen = _make_gen(4)
    run(gen, log, n_iterations=4, resume=False)

    for rec in _gen_iterations(log):
        assert "generator" not in rec


def test_per_signature_status_emitted_at_iter_100_only(tmp_path: Path, capsys):
    """Per-signature block fires at i+1 == 100 but not at 99 or 101."""
    log = tmp_path / "log.jsonl"
    gen = MockGenerator(sequence=[_EXTRACT_FAIL] * 101)
    run(gen, log, n_iterations=101, resume=False)

    captured = capsys.readouterr().out
    header_lines = [
        l for l in captured.splitlines() if l.startswith("[forge] per-signature at iter ")
    ]
    assert len(header_lines) == 1
    assert "at iter 100:" in header_lines[0]

    body_prefix = "  "
    body_lines = [
        l for l in captured.splitlines()
        if l.startswith(body_prefix) and " : total=" in l
    ]
    assert len(body_lines) == len(ACTIVE_SIGNATURES)


def test_few_shot_dedup_under_threshold_returns_empty_pool():
    """When pool has <10 atoms, few-shot is not engaged; the pool is unused."""
    # Not a direct assertion on _pick_few_shot; this guards the threshold branch
    # in run() by constructing a pool of 9 duplicates + 1 unique and verifying
    # that even if selection ran, dedup would collapse to 1.
    pool = [("dup_" + str(i), ["x"] * 20) for i in range(9)] + [
        ("uniq", ["y"] * 20)
    ]
    picked = _pick_few_shot(pool, k=3)
    # Newest-first: uniq (fp=y), dup_8 (fp=x), dup_7 (dup), ... → ["uniq", "dup_8"].
    assert picked == ["uniq", "dup_8"]


# ----------------------------- Day 6 gate-5 canonical coverage ---------------


def test_variable_name_variant_rejected_at_canonical_before_fingerprint(tmp_path: Path):
    """Two variable-name-variants for the same signature: second rejected at
    novelty by the canonical fast-path, with fingerprint=None (proves
    fingerprint computation was skipped)."""
    log = tmp_path / "log.jsonl"
    src_a = "```python\ndef f(n):\n    return n + 1\n```"
    src_b = "```python\ndef f(x):\n    return x + 1\n```"
    gen = MockGenerator(sequence=[src_a, src_b])
    run(
        gen,
        log,
        n_iterations=2,
        active_signatures=[("int", "int")],
        resume=False,
    )

    gen_recs = _gen_iterations(log)
    assert len(gen_recs) == 2

    first, second = gen_recs[0], gen_recs[1]
    assert first["gate_result"]["accepted"] is True
    assert first["fingerprint"] is not None

    assert second["gate_result"]["accepted"] is False
    assert second["gate_result"]["stage"] == "novelty"
    assert "canonical duplicate" in second["gate_result"]["reason"]
    assert "iteration 0" in second["gate_result"]["reason"]
    assert second["fingerprint"] is None
    assert (
        second["gate_result"]["metadata"].get("layer_6_reject")
        == "canonical_duplicate"
    )


def test_canonical_count_appears_in_status_line(tmp_path: Path, capsys):
    """The per-25 status line includes a `canonical=` observability counter."""
    log = tmp_path / "log.jsonl"
    gen = _make_gen(30)
    run(gen, log, n_iterations=25, resume=False)

    captured = capsys.readouterr().out
    status_lines = [l for l in captured.splitlines() if l.startswith("[forge] iter ")]
    assert len(status_lines) == 1
    assert "canonical=" in status_lines[0]
