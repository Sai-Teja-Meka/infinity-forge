"""AST canonicalization for fast cross-variable-name dedup.

Two functions that differ only in variable names (``def f(lst): ...`` vs
``def f(nums): ...``) produce identical canonical forms, so a hash of the
canonical text is a reliable pre-filter before the expensive behavioral
fingerprint. Genuinely different functions produce different canonical
forms; builtins and attribute names are preserved because they carry
semantic meaning.

This module never mutates cascade decisions. It is consulted by the
forge loop after ``cascade.evaluate`` returns accepted, as a fast-path
before the Layer-6 fingerprint.
"""
from __future__ import annotations

import ast
import hashlib

from infinity_forge.sandbox import _ALLOWED_BUILTIN_NAMES


_FUNC_NAME = "f"


def canonicalize(source: str) -> str:
    """Alpha-rename parameter and locals in ``source`` to canonical forms.

    - The single function parameter becomes ``_p``.
    - Local variables (including comprehension and lambda targets) become
      ``_v0``, ``_v1``, ``_v2``, ... in order of first appearance during
      ``ast.walk``.
    - Allowed builtins and the function name ``f`` are preserved.
    - Attribute names are preserved (only the object being accessed may
      be renamed, never the attribute).

    Returns ``ast.unparse()`` of the rewritten tree. If parsing fails,
    returns the original source unchanged.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    func = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func = node
            break
    if func is None:
        return source
    if not func.args.args:
        return source

    param_original = func.args.args[0].arg
    rename_map: dict[str, str] = {param_original: "_p"}
    local_counter = 0

    def _register(name: str) -> None:
        nonlocal local_counter
        if name in rename_map:
            return
        if name in _ALLOWED_BUILTIN_NAMES:
            return
        if name == _FUNC_NAME:
            return
        rename_map[name] = f"_v{local_counter}"
        local_counter += 1

    for node in ast.walk(func):
        if isinstance(node, ast.Name):
            _register(node.id)
        elif isinstance(node, ast.arg):
            _register(node.arg)

    for node in ast.walk(func):
        if isinstance(node, ast.Name):
            if node.id in rename_map:
                node.id = rename_map[node.id]
        elif isinstance(node, ast.arg):
            if node.arg in rename_map:
                node.arg = rename_map[node.arg]

    return ast.unparse(tree)


def canonical_key(canonical_source: str, signature: tuple[str, str]) -> str:
    """SHA-256 hex digest keyed by both signature and canonical source.

    Including the signature prevents cross-signature collapse: the same
    canonical body under a different (input_type, output_type) pair has
    different probe inputs and may yield a different behavioral
    fingerprint, so it must remain a distinct library entry.
    """
    payload = f"{signature}|{canonical_source}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
