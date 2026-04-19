"""Deterministic input samplers for Day 3 generator loop.

Every sampler returns JSON-serializable values (sandbox requirement). Same
`seed` always produces the same value, so iteration indices are reproducible.

Low seed indices (0..4) are reserved for canonical edge cases per type so that
boundary behavior is exercised early in every run.
"""

import random
import string
from typing import Any

_INT_EDGES: list[int] = [0, 1, -1, 2, 10]
_FLOAT_EDGES: list[float] = [0.0, 1.0, -1.0, 0.5, -0.5]
_STR_EDGES: list[str] = ["", "a", "hello", "AbC", "hi world"]
_BOOL_EDGES: list[bool] = [True, False]


def _sample_int(rng: random.Random) -> int:
    return rng.randint(-100, 100)


def _sample_float(rng: random.Random) -> float:
    return round(rng.uniform(-100.0, 100.0), 4)


def _sample_bool(rng: random.Random) -> bool:
    return rng.random() < 0.5


def _sample_str(rng: random.Random) -> str:
    length = rng.randint(0, 20)
    alphabet = string.ascii_letters + string.digits + " "
    return "".join(rng.choice(alphabet) for _ in range(length))


def _sample_list_int(rng: random.Random) -> list[int]:
    length = rng.randint(0, 20)
    return [rng.randint(-100, 100) for _ in range(length)]


def _sample_list_float(rng: random.Random) -> list[float]:
    length = rng.randint(0, 20)
    return [round(rng.uniform(-100.0, 100.0), 4) for _ in range(length)]


def _sample_list_str(rng: random.Random) -> list[str]:
    length = rng.randint(0, 20)
    return [_sample_str(rng) for _ in range(length)]


def _sample_dict(rng: random.Random) -> dict[str, Any]:
    n_keys = rng.randint(0, 5)
    keys_pool = ["a", "b", "c", "d", "e", "f", "g", "h"]
    keys = rng.sample(keys_pool, n_keys)
    result: dict[str, Any] = {}
    for k in keys:
        if rng.random() < 0.5:
            result[k] = rng.randint(-100, 100)
        else:
            result[k] = _sample_str(rng)
    return result


_SAMPLERS = {
    "int": _sample_int,
    "float": _sample_float,
    "bool": _sample_bool,
    "str": _sample_str,
    "list[int]": _sample_list_int,
    "list[float]": _sample_list_float,
    "list[str]": _sample_list_str,
    "dict": _sample_dict,
}


def sample_input(input_type: str, seed: int) -> Any:
    """Return a deterministic JSON-serializable value for `input_type`.

    Seeds 0..4 return canonical edge cases for scalar types so boundary
    behavior is exercised within the first few iterations per signature.
    """
    if input_type not in _SAMPLERS:
        raise ValueError(f"unknown input_type: {input_type!r}")

    if 0 <= seed < 5:
        if input_type == "int":
            return _INT_EDGES[seed]
        if input_type == "float":
            return _FLOAT_EDGES[seed]
        if input_type == "str":
            return _STR_EDGES[seed]
        if input_type == "bool":
            return _BOOL_EDGES[seed % 2]

    rng = random.Random(seed)
    return _SAMPLERS[input_type](rng)
