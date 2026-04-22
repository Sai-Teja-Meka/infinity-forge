"""Tests for canonical.py — AST canonicalization for fast dedup.

Covers the eight cases from the Day 6 spec:
(a) parameter renaming,
(b) comprehension/lambda variable renaming,
(c) builtin names preserved,
(d) attribute names preserved,
(e) variable-name-variant functions collapse,
(f) genuinely different functions do not collapse,
(g) ``canonical_key`` includes the signature,
(h) malformed source returns original unchanged.
"""
from __future__ import annotations

from infinity_forge.canonical import canonical_key, canonicalize
from infinity_forge.sandbox import _ALLOWED_BUILTIN_NAMES


def test_parameter_is_renamed_to_underscore_p():
    src = "def f(lst):\n    return lst"
    out = canonicalize(src)
    assert "_p" in out
    assert "lst" not in out


def test_comprehension_target_is_renamed():
    src = "def f(lst):\n    return [abs(x) for x in lst]"
    out = canonicalize(src)
    assert "x" not in out.split("f(")[1]
    assert "_v0" in out
    assert "_p" in out


def test_lambda_parameter_is_renamed():
    src = "def f(lst):\n    return sorted(lst, key=lambda y: -y)"
    out = canonicalize(src)
    assert " y" not in out
    assert "_v0" in out or "_p" in out


def test_builtin_names_are_preserved():
    src = (
        "def f(lst):\n"
        "    return sum(map(abs, sorted(lst, reverse=True)))"
    )
    out = canonicalize(src)
    for name in ("sum", "map", "abs", "sorted"):
        assert name in out, f"builtin {name!r} lost during canonicalization"


def test_every_allowed_builtin_survives_canonicalization():
    for name in _ALLOWED_BUILTIN_NAMES:
        if name in ("True", "False", "None"):
            continue
        src = f"def f(x):\n    return {name}"
        out = canonicalize(src)
        assert name in out, f"builtin {name!r} was renamed"


def test_attribute_names_are_preserved():
    src = "def f(s):\n    return s.upper()"
    out = canonicalize(src)
    assert ".upper" in out
    assert "_p.upper" in out


def test_attribute_chain_preserved_while_object_renamed():
    src = "def f(d):\n    return sum(d.values())"
    out = canonicalize(src)
    assert ".values" in out
    assert "_p.values" in out


def test_variable_name_variants_produce_same_canonical_form():
    a = "def f(lst):\n    return sum(lst) / len(lst) if lst else 0.0"
    b = "def f(nums):\n    return sum(nums) / len(nums) if nums else 0.0"
    c = "def f(xs):\n    return sum(xs) / len(xs) if xs else 0.0"
    assert canonicalize(a) == canonicalize(b) == canonicalize(c)


def test_local_variable_variants_also_collapse():
    a = "def f(lst):\n    total = sum(lst)\n    return total"
    b = "def f(lst):\n    s = sum(lst)\n    return s"
    assert canonicalize(a) == canonicalize(b)


def test_genuinely_different_functions_differ():
    a = "def f(lst):\n    return sum(lst)"
    b = "def f(lst):\n    return max(lst)"
    assert canonicalize(a) != canonicalize(b)


def test_different_bodies_with_identical_param_still_differ():
    a = "def f(n):\n    return n * 2"
    b = "def f(n):\n    return n + 2"
    assert canonicalize(a) != canonicalize(b)


def test_canonical_key_differs_when_signature_differs():
    canonical = canonicalize("def f(lst):\n    return sum(lst)")
    k1 = canonical_key(canonical, ("list[int]", "int"))
    k2 = canonical_key(canonical, ("list[float]", "float"))
    assert k1 != k2


def test_canonical_key_matches_when_both_canonical_and_signature_match():
    canonical = canonicalize("def f(lst):\n    return sum(lst)")
    sig = ("list[int]", "int")
    assert canonical_key(canonical, sig) == canonical_key(canonical, sig)


def test_canonical_key_is_sha256_hex():
    k = canonical_key("def f(_p):\n    return _p", ("int", "int"))
    assert len(k) == 64
    assert all(c in "0123456789abcdef" for c in k)


def test_malformed_source_returns_original_unchanged():
    bad = "def f(x:\n    return x"  # intentional syntax error
    assert canonicalize(bad) == bad


def test_non_function_source_returns_original_unchanged():
    not_a_func = "x = 1\ny = 2"
    assert canonicalize(not_a_func) == not_a_func
