"""Hand-written seed atoms for signatures the generator struggles with.

The Day 3 run showed two signatures with zero or near-zero accepted atoms:
``int -> list[int]`` and ``dict -> int``. Without any accepted atom in a
signature's library, the few-shot pool is empty, so every regeneration
starts cold from the same prompt template and the same mistakes repeat.

These seeds break that cycle. They are injected at forge initialization
(iteration numbers -1, -2, -3, ...) so the library has something to pull
for few-shot examples from turn 0 onward.

Seeds are NOT trusted blindly: they go through the same cascade as
generator output. Any seed using a banned builtin (e.g. ``isinstance``)
is rejected by Layer 2 just like a generator candidate would be. Rewrite
the seed; do not widen the cascade.
"""
from __future__ import annotations

SEED_ATOMS: list[dict] = [
    {
        "signature": ("int", "list[int]"),
        "source": "def f(n):\n    return [i for i in range(1, abs(n) + 1) if n % i == 0]",
        "note": "divisors of abs(n); handles n=0 by returning []",
    },
    {
        "signature": ("dict", "int"),
        "source": "def f(d):\n    return sum(len(str(v)) for v in d.values())",
        "note": "sum of string-lengths of dict values; avoids isinstance",
    },
    {
        "signature": ("list[str]", "list[str]"),
        "source": "def f(lst):\n    return sorted(lst, key=len)",
        "note": "sort strings by length",
    },
    # Day 9 bridge seeds: one atom per new bridge signature so every
    # cluster has a non-empty few-shot pool and the type graph is
    # strongly connected from turn 0.
    {
        "signature": ("int", "str"),
        "source": "def f(n):\n    return str(abs(n))",
        "note": "string form of abs(n); bridges int -> str cluster",
    },
    {
        "signature": ("str", "list[str]"),
        "source": "def f(s):\n    return list(s)",
        "note": "characters of s as a list; enables str -> list[str] -> str chains",
    },
    {
        "signature": ("str", "list[int]"),
        "source": "def f(s):\n    return [len(w) for w in s.split()]",
        "note": "word-length list; bridges str into list[int] cluster",
    },
    {
        "signature": ("list[int]", "str"),
        "source": "def f(lst):\n    return ''.join(str(x) for x in lst)",
        "note": "concatenated decimal repr; bridges list[int] back to str",
    },
    {
        "signature": ("bool", "int"),
        "source": "def f(b):\n    return 1 if b else 0",
        "note": "bool -> {0,1}; makes bool non-terminal",
    },
    {
        "signature": ("int", "dict"),
        "source": "def f(n):\n    return {str(i): i * i for i in range(abs(n) % 10)}",
        "note": "squares map capped at 9 entries; gives dict a second entry point",
    },
]
