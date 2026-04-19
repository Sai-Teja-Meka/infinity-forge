"""Gate 1 tests: probes.py is structurally correct and pins the canonical edges."""
from __future__ import annotations

import json

from infinity_forge.probes import EXPECTED_OUTPUT_TYPES, PROBES

ALL_INPUT_TYPES = [
    "int",
    "float",
    "bool",
    "str",
    "list[int]",
    "list[float]",
    "list[str]",
    "dict",
]

ALL_OUTPUT_TYPES = [
    "int",
    "float",
    "bool",
    "str",
    "list[int]",
    "list[float]",
    "list[str]",
    "dict",
]


def test_probes_has_all_eight_input_types():
    assert set(PROBES.keys()) == set(ALL_INPUT_TYPES)


def test_probes_each_type_has_exactly_twenty_values():
    for input_type in ALL_INPUT_TYPES:
        assert len(PROBES[input_type]) == 20, (
            f"expected 20 probes for {input_type}, got {len(PROBES[input_type])}"
        )


def test_int_first_five_edges():
    assert PROBES["int"][:5] == [0, 1, -1, 2, 10]


def test_float_first_five_edges():
    assert PROBES["float"][:5] == [0.0, 1.0, -1.0, 0.5, -0.5]


def test_str_first_five_edges():
    assert PROBES["str"][:5] == ["", "a", "hello", "AbC", "hi world"]


def test_bool_first_five_edges():
    assert PROBES["bool"][:5] == [True, False, True, False, True]


def test_every_probe_value_is_json_serializable():
    for input_type, values in PROBES.items():
        for i, value in enumerate(values):
            try:
                json.dumps(value)
            except (TypeError, ValueError) as e:
                raise AssertionError(
                    f"PROBES[{input_type!r}][{i}] is not JSON-serializable: {e}"
                )


def test_expected_output_types_has_all_eight():
    assert set(EXPECTED_OUTPUT_TYPES.keys()) == set(ALL_OUTPUT_TYPES)


def test_expected_output_types_are_shallow_python_types():
    assert EXPECTED_OUTPUT_TYPES["int"] is int
    assert EXPECTED_OUTPUT_TYPES["float"] is float
    assert EXPECTED_OUTPUT_TYPES["bool"] is bool
    assert EXPECTED_OUTPUT_TYPES["str"] is str
    assert EXPECTED_OUTPUT_TYPES["list[int]"] is list
    assert EXPECTED_OUTPUT_TYPES["list[float]"] is list
    assert EXPECTED_OUTPUT_TYPES["list[str]"] is list
    assert EXPECTED_OUTPUT_TYPES["dict"] is dict
