"""Tests for inputs.py."""

import json

import pytest

from infinity_forge.inputs import sample_input
from infinity_forge.signatures import INPUT_TYPES

ALL_TYPES = INPUT_TYPES


@pytest.mark.parametrize("t", ALL_TYPES)
def test_sampler_determinism(t):
    a = sample_input(t, seed=42)
    b = sample_input(t, seed=42)
    assert a == b


@pytest.mark.parametrize("t", ALL_TYPES)
def test_sampler_json_serializable(t):
    for seed in range(0, 30):
        v = sample_input(t, seed=seed)
        json.dumps(v)  # raises if not serializable


@pytest.mark.parametrize("t", ALL_TYPES)
def test_every_type_is_covered(t):
    v = sample_input(t, seed=0)
    assert v is not None or t == "int"  # int seed=0 is 0 which is not None anyway


def test_unknown_type_raises():
    with pytest.raises(ValueError):
        sample_input("complex", seed=0)


def test_tuple_not_supported():
    with pytest.raises(ValueError):
        sample_input("tuple", seed=0)


def test_int_edge_cases_early():
    early = [sample_input("int", seed=i) for i in range(5)]
    assert 0 in early
    assert 1 in early
    assert -1 in early


def test_float_edge_cases_early():
    early = [sample_input("float", seed=i) for i in range(5)]
    assert 0.0 in early
    assert 1.0 in early
    assert -1.0 in early


def test_str_edge_cases_early():
    early = [sample_input("str", seed=i) for i in range(5)]
    assert "" in early
    assert any(len(s) == 1 for s in early)


def test_bool_edge_cases_early():
    early = [sample_input("bool", seed=i) for i in range(5)]
    assert True in early
    assert False in early


def test_int_value_type():
    for seed in range(5, 20):
        v = sample_input("int", seed=seed)
        assert isinstance(v, int) and not isinstance(v, bool)


def test_float_value_type():
    for seed in range(5, 20):
        v = sample_input("float", seed=seed)
        assert isinstance(v, float)


def test_bool_value_type():
    for seed in range(20):
        v = sample_input("bool", seed=seed)
        assert isinstance(v, bool)


def test_str_value_type():
    for seed in range(20):
        v = sample_input("str", seed=seed)
        assert isinstance(v, str)
        assert len(v) <= 20


def test_list_int_value_type():
    for seed in range(20):
        v = sample_input("list[int]", seed=seed)
        assert isinstance(v, list)
        assert len(v) <= 20
        for x in v:
            assert isinstance(x, int) and not isinstance(x, bool)


def test_list_float_value_type():
    for seed in range(20):
        v = sample_input("list[float]", seed=seed)
        assert isinstance(v, list)
        assert len(v) <= 20
        for x in v:
            assert isinstance(x, float)


def test_list_str_value_type():
    for seed in range(20):
        v = sample_input("list[str]", seed=seed)
        assert isinstance(v, list)
        assert len(v) <= 20
        for x in v:
            assert isinstance(x, str)


def test_dict_value_type():
    for seed in range(20):
        v = sample_input("dict", seed=seed)
        assert isinstance(v, dict)
        assert len(v) <= 5
        for k in v:
            assert isinstance(k, str)


def test_different_seeds_differ_sometimes():
    # not a strict invariant, but across 20 seeds we expect diversity
    vals = [sample_input("int", seed=i) for i in range(5, 25)]
    assert len(set(vals)) > 1
