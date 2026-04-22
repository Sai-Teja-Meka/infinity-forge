"""Tests for signatures.py."""

import pytest

from infinity_forge.signatures import (
    ACTIVE_SIGNATURES,
    INPUT_TYPES,
    OUTPUT_TYPES,
    describe_type,
)


def test_input_types_count():
    assert len(INPUT_TYPES) == 8


def test_output_types_count():
    assert len(OUTPUT_TYPES) == 8


def test_active_signatures_count():
    assert len(ACTIVE_SIGNATURES) == 14


def test_no_tuple_anywhere():
    assert "tuple" not in INPUT_TYPES
    assert "tuple" not in OUTPUT_TYPES
    for inp, out in ACTIVE_SIGNATURES:
        assert inp != "tuple" and out != "tuple"


def test_active_signatures_use_declared_types():
    for inp, out in ACTIVE_SIGNATURES:
        assert inp in INPUT_TYPES, f"{inp} not in INPUT_TYPES"
        assert out in OUTPUT_TYPES, f"{out} not in OUTPUT_TYPES"


def test_active_signatures_unique():
    assert len(set(ACTIVE_SIGNATURES)) == len(ACTIVE_SIGNATURES)


def test_input_types_unique():
    assert len(set(INPUT_TYPES)) == len(INPUT_TYPES)


def test_output_types_unique():
    assert len(set(OUTPUT_TYPES)) == len(OUTPUT_TYPES)


@pytest.mark.parametrize("type_str", [
    "int", "float", "bool", "str",
    "list[int]", "list[float]", "list[str]", "dict",
])
def test_describe_type_covers_all_inputs(type_str):
    desc = describe_type(type_str)
    assert isinstance(desc, str)
    assert len(desc) > 0


def test_describe_type_unknown_raises():
    with pytest.raises(ValueError):
        describe_type("complex")


def test_describe_type_tuple_rejected():
    with pytest.raises(ValueError):
        describe_type("tuple")


def test_describe_type_content():
    assert "integer" in describe_type("int")
    assert "float" in describe_type("float")
    assert "boolean" in describe_type("bool")
    assert "string" in describe_type("str")
    assert "list" in describe_type("list[int]")
    assert "dict" in describe_type("dict").lower()
