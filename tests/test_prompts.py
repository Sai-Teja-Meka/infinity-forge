"""Tests for prompts.py."""

import pytest

from infinity_forge.prompts import build_prompt
from infinity_forge.sandbox import _ALLOWED_BUILTIN_NAMES
from infinity_forge.signatures import ACTIVE_SIGNATURES, describe_type


def test_prompt_contains_input_description():
    p = build_prompt("int", "int")
    assert describe_type("int") in p


def test_prompt_contains_output_description():
    p = build_prompt("list[int]", "dict")
    assert describe_type("list[int]") in p
    assert describe_type("dict") in p


def test_prompt_contains_raw_type_strings():
    p = build_prompt("list[str]", "str")
    assert "list[str]" in p
    assert "str" in p


def test_prompt_lists_all_26_builtins():
    p = build_prompt("int", "int")
    assert len(_ALLOWED_BUILTIN_NAMES) == 26
    for name in _ALLOWED_BUILTIN_NAMES:
        assert name in p, f"builtin {name!r} missing from prompt"


def test_prompt_mentions_no_imports():
    p = build_prompt("int", "int")
    assert "No imports" in p or "no imports" in p.lower()


def test_prompt_mentions_max_30_lines():
    p = build_prompt("int", "int")
    assert "30 lines" in p


def test_prompt_mentions_complexity():
    p = build_prompt("int", "int")
    assert "complexity" in p.lower()


def test_few_shot_none_excludes_section():
    p = build_prompt("int", "int", few_shot_atoms=None)
    assert "Examples:" not in p


def test_few_shot_empty_excludes_section():
    p = build_prompt("int", "int", few_shot_atoms=[])
    assert "Examples:" not in p


def test_few_shot_included_when_provided():
    atoms = ["def f(x):\n    return x + 1"]
    p = build_prompt("int", "int", few_shot_atoms=atoms)
    assert "Examples:" in p
    assert "def f(x):" in p
    assert "return x + 1" in p


def test_few_shot_caps_at_three():
    atoms = [f"def f(x):\n    return x + {i}" for i in range(10)]
    p = build_prompt("int", "int", few_shot_atoms=atoms)
    assert "return x + 0" in p
    assert "return x + 1" in p
    assert "return x + 2" in p
    assert "return x + 3" not in p


def test_prompt_ends_with_call_to_action():
    p = build_prompt("int", "int")
    assert p.rstrip().endswith("Now write your function:")


def test_prompt_mentions_function_name_f():
    p = build_prompt("int", "int")
    assert "`f`" in p or " f " in p or "named f" in p


def test_prompt_mentions_one_positional_argument():
    p = build_prompt("int", "int")
    assert "one positional" in p.lower()


def test_prompt_tells_model_not_to_validate_input():
    p = build_prompt("int", "int")
    assert (
        "The input is guaranteed to match the declared type. "
        "Do not validate or check the input — write the function body "
        "assuming correct input."
    ) in p


def test_no_validate_sentence_appears_before_constraint_list():
    p = build_prompt("list[int]", "bool")
    no_validate_idx = p.index("Do not validate")
    builtins_idx = p.index("Allowed builtins only")
    assert no_validate_idx < builtins_idx


def test_no_validate_sentence_appears_after_return_type_line():
    p = build_prompt("int", "int")
    returns_idx = p.index("It returns")
    no_validate_idx = p.index("Do not validate")
    assert returns_idx < no_validate_idx


def test_prompt_forbids_isinstance_by_name():
    p = build_prompt("int", "int")
    assert "Never call isinstance — it is not available and will cause rejection." in p


def test_isinstance_sentence_appears_immediately_after_do_not_validate_sentence():
    p = build_prompt("dict", "int")
    no_validate_idx = p.index("Do not validate")
    isinstance_idx = p.index("Never call isinstance")
    builtins_idx = p.index("Allowed builtins only")
    assert no_validate_idx < isinstance_idx < builtins_idx


def test_prompt_mentions_edge_case_hardening():
    p = build_prompt("int", "int")
    assert (
        "Your function may receive empty lists, empty strings, empty dicts, "
        "zero, negative numbers, or single-element inputs. Handle edge cases "
        "gracefully — never raise an exception."
    ) in p


def test_edge_case_sentence_appears_after_isinstance_and_before_builtins():
    p = build_prompt("list[int]", "int")
    isinstance_idx = p.index("Never call isinstance")
    edge_idx = p.index("Your function may receive empty lists")
    builtins_idx = p.index("Allowed builtins only")
    assert isinstance_idx < edge_idx < builtins_idx


@pytest.mark.parametrize("inp,out", ACTIVE_SIGNATURES)
def test_prompt_builds_for_every_active_signature(inp, out):
    p = build_prompt(inp, out)
    assert isinstance(p, str)
    assert len(p) > 0


def test_prompt_is_reasonably_sized():
    p = build_prompt("int", "int")
    # ~800 token budget; 4 chars/token heuristic → well under 3200 chars
    assert len(p) < 3200
