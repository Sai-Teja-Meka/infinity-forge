"""Integration tests for forge.run(). MockGenerator only — no real LLM."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from infinity_forge.forge import run
from infinity_forge.generator import MockGenerator
from infinity_forge.signatures import ACTIVE_SIGNATURES


def _sig_for(iteration: int) -> tuple[str, str]:
    return ACTIVE_SIGNATURES[iteration % len(ACTIVE_SIGNATURES)]


def _temp_for(iteration: int) -> float:
    return (0.7, 0.9, 1.1)[iteration % 3]


_ACCEPT_SRC = "```python\ndef f(x):\n    y = 1\n    return y + 1\n```"
_REJECT_SRC = "```python\ndef f(x):\n    return open('/etc/passwd').read()\n```"
_EXTRACT_FAIL = "no code here at all, just prose"


def _build_sequence(n: int) -> list[str]:
    """Per-iteration response list.

    Over the first 20 iterations, i % 10 selects the bucket:
      - i % 10 in {0, 1}:          accepted  (8 total)
      - i % 10 in {2, 3, 4}:       cascade-rejected (6 total)
      - i % 10 in {5, 6, 7}:       extract_code returns None (6 total)
      - i % 10 in {8, 9}:          accepted  (4 extra to hit 8 accepts)
    Distribution across i=0..19: 8 accepted, 6 rejected, 6 extract-fail.
    """
    out: list[str] = []
    for i in range(n):
        b = i % 10
        if b in (0, 1, 8, 9):
            out.append(_ACCEPT_SRC)
        elif b in (2, 3, 4):
            out.append(_REJECT_SRC)
        else:
            out.append(_EXTRACT_FAIL)
    return out


def _make_gen(n: int) -> MockGenerator:
    return MockGenerator(sequence=_build_sequence(n))


def test_run_writes_jsonl_with_expected_line_count(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    gen = _make_gen(20)
    run(gen, log, n_iterations=20, resume=False)

    lines = log.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 20


def test_every_line_is_valid_json_with_expected_fields(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    gen = _make_gen(20)
    run(gen, log, n_iterations=20, resume=False)

    expected_fields = {
        "iteration", "signature", "temperature", "raw_llm_output",
        "extracted_source", "input_value", "gate_result", "timestamp",
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

    for line in log.read_text(encoding="utf-8").strip().split("\n"):
        rec = json.loads(line)
        i = rec["iteration"]
        expected_sig = list(_sig_for(i))
        assert rec["signature"] == expected_sig
        assert rec["temperature"] == _temp_for(i)


def test_acceptance_and_rejection_counts(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    gen = _make_gen(20)
    run(gen, log, n_iterations=20, resume=False)

    accepted = 0
    layer2_rejected = 0
    extract_failed = 0
    for line in log.read_text(encoding="utf-8").strip().split("\n"):
        rec = json.loads(line)
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
    # Shared generator so the sequence continues across the two run() calls.
    gen = _make_gen(20)

    run(gen, log, n_iterations=10, resume=False)
    lines_after_first = log.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines_after_first) == 10
    first_iters = [json.loads(l)["iteration"] for l in lines_after_first]
    assert first_iters == list(range(10))

    run(gen, log, n_iterations=10, resume=True)
    lines_all = log.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines_all) == 20
    all_iters = [json.loads(l)["iteration"] for l in lines_all]
    assert all_iters == list(range(20))


def test_resumption_no_overlap(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    gen = _make_gen(20)
    run(gen, log, n_iterations=10, resume=False)
    run(gen, log, n_iterations=10, resume=True)

    iters = [json.loads(l)["iteration"] for l in log.read_text().strip().split("\n")]
    assert len(iters) == len(set(iters))


def test_status_line_printed_every_25(tmp_path: Path, capsys):
    log = tmp_path / "log.jsonl"
    gen = _make_gen(60)
    run(gen, log, n_iterations=50, resume=False)
    captured = capsys.readouterr().out
    status_lines = [l for l in captured.splitlines() if l.startswith("[forge]")]
    # Iterations 0..49 → status at end of iter 24 and iter 49
    assert len(status_lines) == 2


def test_status_line_not_printed_below_25(tmp_path: Path, capsys):
    log = tmp_path / "log.jsonl"
    gen = _make_gen(20)
    run(gen, log, n_iterations=20, resume=False)
    captured = capsys.readouterr().out
    status_lines = [l for l in captured.splitlines() if l.startswith("[forge]")]
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
    iters = [json.loads(l)["iteration"] for l in log.read_text().strip().split("\n")]
    # Second run starts at 0 again (no resume), so iterations overlap.
    assert iters == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
