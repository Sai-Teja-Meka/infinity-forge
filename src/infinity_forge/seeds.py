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
    {
        "signature": ("list[float]", "float"),
        "source": "def f(lst):\n    return sum(lst) / len(lst) if lst else 0.0",
        "note": "mean of list; returns float via true division; handles empty list",
    },
]
