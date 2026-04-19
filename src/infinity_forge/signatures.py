"""Type signatures used by the Day 3 generator loop.

Eight input types × eight output types form a 64-cell grid; Day 3 activates a
practical subset of 15 cells that pure-stdlib Python functions handle well.

Tuple support is deferred until IPC round-tripping is addressed (JSON has no
tuple type, so feeding a list to a tuple-typed function would fail to exercise
tuple semantics).
"""

INPUT_TYPES: list[str] = [
    "int",
    "float",
    "bool",
    "str",
    "list[int]",
    "list[float]",
    "list[str]",
    "dict",
]

OUTPUT_TYPES: list[str] = [
    "int",
    "float",
    "bool",
    "str",
    "list[int]",
    "list[float]",
    "list[str]",
    "dict",
]

ACTIVE_SIGNATURES: list[tuple[str, str]] = [
    ("int", "int"),
    ("int", "bool"),
    ("int", "list[int]"),
    ("list[int]", "int"),
    ("list[int]", "bool"),
    ("list[int]", "list[int]"),
    ("list[int]", "dict"),
    ("list[float]", "float"),
    ("list[str]", "str"),
    ("list[str]", "list[str]"),
    ("str", "int"),
    ("str", "bool"),
    ("str", "str"),
    ("dict", "int"),
    ("dict", "list[str]"),
]

_DESCRIPTIONS: dict[str, str] = {
    "int": "an integer",
    "float": "a float",
    "bool": "a boolean",
    "str": "a string",
    "list[int]": "a list of integers",
    "list[float]": "a list of floats",
    "list[str]": "a list of strings",
    "dict": "a dictionary with string keys",
}


def describe_type(type_str: str) -> str:
    """Return a human-readable description of a type string."""
    if type_str not in _DESCRIPTIONS:
        raise ValueError(f"unknown type: {type_str!r}")
    return _DESCRIPTIONS[type_str]
