"""Gate 2 tests: every seed atom passes the cascade and executes cleanly.

Seeds are hand-written but not trusted — they must survive Layer 1 +
Layer 2 (cascade.evaluate) just like generator output, and they must
produce an ok sandbox execution on at least one representative input.
"""
from __future__ import annotations

import pytest

from infinity_forge import cascade, sandbox
from infinity_forge.seeds import SEED_ATOMS


REPRESENTATIVE_INPUTS: dict[tuple[str, str], object] = {
    ("int", "list[int]"): 12,
    ("dict", "int"): {"a": 1, "b": "hello"},
    ("list[str]", "list[str]"): ["bb", "a", "ccc"],
}


def test_seed_atoms_nonempty():
    assert len(SEED_ATOMS) >= 3


def test_seed_atoms_cover_the_three_weak_signatures():
    sigs = {tuple(s["signature"]) for s in SEED_ATOMS}
    assert ("int", "list[int]") in sigs
    assert ("dict", "int") in sigs
    assert ("list[str]", "list[str]") in sigs


def test_every_seed_has_required_fields():
    for seed in SEED_ATOMS:
        assert "signature" in seed
        assert "source" in seed
        assert "note" in seed
        assert isinstance(seed["signature"], tuple)
        assert len(seed["signature"]) == 2
        assert isinstance(seed["source"], str)
        assert seed["source"].strip()


@pytest.mark.parametrize("seed", SEED_ATOMS, ids=lambda s: f"{s['signature'][0]}->{s['signature'][1]}")
def test_seed_passes_cascade(seed):
    result = cascade.evaluate(seed["source"])
    assert result["accepted"] is True, (
        f"seed {seed['signature']} rejected at {result['stage']}: {result['reason']}"
    )
    assert result["stage"] == "layer_2"


@pytest.mark.parametrize("seed", SEED_ATOMS, ids=lambda s: f"{s['signature'][0]}->{s['signature'][1]}")
def test_seed_executes_in_sandbox(seed):
    sig = tuple(seed["signature"])
    if sig not in REPRESENTATIVE_INPUTS:
        pytest.skip(f"no representative input registered for {sig}")
    input_value = REPRESENTATIVE_INPUTS[sig]
    result = sandbox.run_in_sandbox(seed["source"], input_value)
    assert result["status"] == "ok", (
        f"seed {sig} sandbox status={result['status']}: {result}"
    )
    assert "output" in result


def test_int_to_list_int_seed_correct_on_12():
    seed = next(s for s in SEED_ATOMS if s["signature"] == ("int", "list[int]"))
    result = sandbox.run_in_sandbox(seed["source"], 12)
    assert result["status"] == "ok"
    assert result["output"] == [1, 2, 3, 4, 6, 12]


def test_int_to_list_int_seed_handles_zero():
    seed = next(s for s in SEED_ATOMS if s["signature"] == ("int", "list[int]"))
    result = sandbox.run_in_sandbox(seed["source"], 0)
    assert result["status"] == "ok"
    assert result["output"] == []


def test_dict_to_int_seed_correct():
    seed = next(s for s in SEED_ATOMS if s["signature"] == ("dict", "int"))
    result = sandbox.run_in_sandbox(seed["source"], {"a": 1, "b": "hello"})
    assert result["status"] == "ok"
    # len(str(1)) = 1, len(str("hello")) = 5
    assert result["output"] == 6


def test_list_str_to_list_str_seed_correct():
    seed = next(s for s in SEED_ATOMS if s["signature"] == ("list[str]", "list[str]"))
    result = sandbox.run_in_sandbox(seed["source"], ["bb", "a", "ccc"])
    assert result["status"] == "ok"
    assert result["output"] == ["a", "bb", "ccc"]
