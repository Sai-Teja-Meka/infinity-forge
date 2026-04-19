"""Day 2 cascade for ∞ Forge.

Two static-analysis layers that run *before* the sandbox is ever spawned,
an ``evaluate`` function that composes them, and a top-level ``gate`` that
runs ``evaluate`` and delegates accepted candidates to the Day 1 sandbox.

- :func:`layer_1_structure` — parse + structural checks (single top-level
  ``FunctionDef``, signature shape, non-trivial body, length cap, cyclomatic
  complexity cap). Returns a raw
  ``tuple[bool, str | None, dict]`` of ``(accepted, reason, metadata)``.
- :func:`layer_2_safety` — name/attribute allowlist walk over the AST.
  Same raw tuple shape as Layer 1. (Gate 4+.)
- :func:`evaluate` — runs Layer 1 then Layer 2 on a source string and
  assembles the layer outputs into a full :class:`Result`. Does not touch
  the sandbox.
- :func:`gate` — calls :func:`evaluate`; on acceptance, delegates to
  :func:`infinity_forge.sandbox.run_in_sandbox` with a single ``input_value``
  and returns the final :class:`Result`.
"""
from __future__ import annotations

import ast
import builtins  # Used for rejection-reason diagnostics only; the security gate is _ALLOWED_BUILTIN_NAMES below.
from typing import Any, Literal, TypedDict

from radon.complexity import cc_visit

from infinity_forge import sandbox
from infinity_forge.sandbox import _ALLOWED_BUILTIN_NAMES

if not isinstance(_ALLOWED_BUILTIN_NAMES, frozenset):
    raise ImportError(
        "infinity_forge.sandbox._ALLOWED_BUILTIN_NAMES must be a frozenset; "
        f"got {type(_ALLOWED_BUILTIN_NAMES).__name__}"
    )


_MAX_LINES = 30
_MAX_COMPLEXITY = 10


class Result(TypedDict):
    accepted: bool
    stage: Literal["layer_1", "layer_2", "sandbox", "completed"]
    reason: str | None
    value: Any | None
    duration_ms: float | None
    metadata: dict


def _extract_names(target: ast.AST, out: set[str]) -> None:
    """Recursively collect Name.id values from an assignment target."""
    if isinstance(target, ast.Name):
        out.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for el in target.elts:
            _extract_names(el, out)
    elif isinstance(target, ast.Starred):
        _extract_names(target.value, out)


def _collect_locals(func_def: ast.FunctionDef) -> set[str]:
    """Collect every name bound inside ``func_def`` (including nested scopes).

    Handles the full spec list: parameters of all nested FunctionDef /
    AsyncFunctionDef / Lambda, Assign.targets, AugAssign/AnnAssign.target,
    For.target, comprehension.target, With.items[*].optional_vars,
    ExceptHandler.name, nested FunctionDef/AsyncFunctionDef names, and
    NamedExpr.target.id. Tuple/List/Starred unpacking is recursed into.
    """
    locals_set: set[str] = {func_def.name}
    for node in ast.walk(func_def):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = node.args
            for arg in a.posonlyargs + a.args + a.kwonlyargs:
                locals_set.add(arg.arg)
            if a.vararg is not None:
                locals_set.add(a.vararg.arg)
            if a.kwarg is not None:
                locals_set.add(a.kwarg.arg)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not func_def:
                locals_set.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                _extract_names(t, locals_set)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            _extract_names(node.target, locals_set)
        elif isinstance(node, ast.For):
            _extract_names(node.target, locals_set)
        elif isinstance(node, ast.comprehension):
            _extract_names(node.target, locals_set)
        elif isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars is not None:
                    _extract_names(item.optional_vars, locals_set)
        elif isinstance(node, ast.ExceptHandler):
            if node.name is not None:
                locals_set.add(node.name)
        elif isinstance(node, ast.NamedExpr):
            if isinstance(node.target, ast.Name):
                locals_set.add(node.target.id)
    return locals_set


def _trivial_reason(func_def: ast.FunctionDef, param_name: str) -> str | None:
    if len(func_def.body) != 1:
        return None
    stmt = func_def.body[0]
    if isinstance(stmt, ast.Pass):
        return "layer_1: trivial pass body"
    if isinstance(stmt, ast.Return):
        val = stmt.value
        if isinstance(val, ast.Name) and val.id == param_name:
            return "layer_1: trivial identity return"
    return None


def layer_1_structure(source: str) -> tuple[bool, str | None, dict]:
    """Layer 1: parse + structural checks.

    Eight rejection rules (in order):
      1. source must parse
      2. exactly one top-level statement
      3. that statement must be a (non-async) ``FunctionDef``
      4. function must take exactly one positional argument
      5. no ``*args`` / ``**kwargs`` / keyword-only / defaults
      6. body must be non-trivial (not ``pass``, not ``return <arg>``)
      7. total source length ≤ 30 physical lines
      8. radon cyclomatic complexity ≤ 10
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, f"layer_1: syntax error: {e.msg}", {"rule": "parse"}

    if len(tree.body) == 0:
        return False, "layer_1: empty source: no top-level statements", {"rule": "single_top_level", "count": 0}
    if len(tree.body) > 1:
        return (
            False,
            f"layer_1: multiple top-level statements (got {len(tree.body)})",
            {"rule": "single_top_level", "count": len(tree.body)},
        )

    node = tree.body[0]

    if isinstance(node, ast.AsyncFunctionDef):
        return False, "layer_1: async functions not allowed", {"rule": "async"}

    if not isinstance(node, ast.FunctionDef):
        return (
            False,
            f"layer_1: top-level statement is not a function (got {type(node).__name__})",
            {"rule": "not_a_function", "node_type": type(node).__name__},
        )

    args = node.args
    if args.vararg is not None:
        return False, "layer_1: *args varargs not allowed", {"rule": "signature", "issue": "vararg"}
    if args.kwarg is not None:
        return False, "layer_1: **kwargs not allowed", {"rule": "signature", "issue": "kwarg"}
    if args.kwonlyargs:
        return False, "layer_1: keyword-only arguments not allowed", {"rule": "signature", "issue": "kwonly"}
    if args.defaults or args.kw_defaults:
        return False, "layer_1: default argument values not allowed", {"rule": "signature", "issue": "default"}

    total_pos = len(args.posonlyargs) + len(args.args)
    if total_pos != 1:
        return (
            False,
            f"layer_1: function must take exactly one positional argument (got {total_pos})",
            {"rule": "signature", "issue": "arg_count", "count": total_pos},
        )

    param_name = (args.posonlyargs + args.args)[0].arg
    trivial = _trivial_reason(node, param_name)
    if trivial is not None:
        return False, trivial, {"rule": "trivial_body"}

    line_count = node.end_lineno - node.lineno + 1
    if line_count > _MAX_LINES:
        return (
            False,
            f"layer_1: function exceeds {_MAX_LINES} lines (got {line_count})",
            {"rule": "length", "lines": line_count},
        )

    blocks = cc_visit(source)
    if not blocks:
        return False, "layer_1: complexity analysis returned no blocks", {"rule": "complexity"}
    complexity = blocks[0].complexity
    if complexity > _MAX_COMPLEXITY:
        return (
            False,
            f"layer_1: cyclomatic complexity exceeds {_MAX_COMPLEXITY} (got {complexity})",
            {"rule": "complexity", "value": complexity},
        )

    return True, None, {"lines": line_count, "complexity": complexity, "param_name": param_name}


def layer_2_safety(source: str) -> tuple[bool, str | None, dict]:
    """Layer 2: name / attribute allowlist walk over the AST.

    Seven rejection rules (in order):
      1. no ``import`` / ``from ... import ...`` statements
      2. no reference to the ``__import__`` name
      3. no dunder attribute access (``x.__class__``, ``x.__dict__``, ...)
      4. no ``global`` / ``nonlocal`` declarations
      5. no ``yield`` / ``yield from``
      6. no ``del`` statement
      7. every Load-context ``Name`` must be either locally bound or in
         ``_ALLOWED_BUILTIN_NAMES``. A name that is a real Python builtin
         but absent from the allowlist is reported as "forbidden builtin";
         otherwise as "undefined name".
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, f"layer_2: syntax error: {e.msg}", {"rule": "parse"}

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return False, "layer_2: expected a single top-level FunctionDef", {"rule": "structure"}

    func_def = tree.body[0]

    for node in ast.walk(func_def):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "layer_2: import statement not allowed", {"rule": "import"}
        if isinstance(node, ast.Name) and node.id == "__import__":
            return False, "layer_2: __import__ not allowed", {"rule": "dunder_import"}
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return (
                False,
                f"layer_2: dunder attribute access not allowed ({node.attr})",
                {"rule": "dunder", "attr": node.attr},
            )
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return False, "layer_2: global/nonlocal not allowed", {"rule": "global"}
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            return False, "layer_2: yield not allowed", {"rule": "yield"}
        if isinstance(node, ast.Delete):
            return False, "layer_2: del statement not allowed", {"rule": "del"}

    locals_set = _collect_locals(func_def)

    for node in ast.walk(func_def):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            name = node.id
            if name in locals_set or name in _ALLOWED_BUILTIN_NAMES:
                continue
            if hasattr(builtins, name):
                return (
                    False,
                    f"layer_2: forbidden builtin '{name}'",
                    {"rule": "forbidden_builtin", "name": name},
                )
            return (
                False,
                f"layer_2: undefined name '{name}'",
                {"rule": "undefined_name", "name": name},
            )

    return True, None, {"locals": sorted(locals_set)}


def evaluate(source: str) -> Result:
    """Run Layer 1 then Layer 2 and assemble a :class:`Result`.

    This function does NOT call the sandbox — ``value`` and ``duration_ms``
    stay ``None`` regardless of outcome. On rejection, ``stage`` identifies
    the layer that rejected; on acceptance, ``stage`` is ``"layer_2"``
    (the last cascade layer reached) and ``metadata`` is the merged dict
    of both layers' metadata.
    """
    l1_accepted, l1_reason, l1_meta = layer_1_structure(source)
    if not l1_accepted:
        return Result(
            accepted=False,
            stage="layer_1",
            reason=l1_reason,
            value=None,
            duration_ms=None,
            metadata=l1_meta,
        )

    l2_accepted, l2_reason, l2_meta = layer_2_safety(source)
    if not l2_accepted:
        return Result(
            accepted=False,
            stage="layer_2",
            reason=l2_reason,
            value=None,
            duration_ms=None,
            metadata=l2_meta,
        )

    return Result(
        accepted=True,
        stage="layer_2",
        reason=None,
        value=None,
        duration_ms=None,
        metadata={**l1_meta, **l2_meta},
    )


def gate(source_code: str, input_value: Any, timeout_ms: int | None = None) -> Result:
    """Run :func:`evaluate`; on acceptance, delegate to :func:`sandbox.run_in_sandbox`.

    Short-circuits on cascade rejection — ``duration_ms`` stays ``None``
    (no subprocess spawned). On acceptance, calls ``run_in_sandbox`` with
    ``timeout_ms`` forwarded only when non-None (otherwise the sandbox's
    own default applies). A ``bad_source`` result from the sandbox is
    treated as a cascade correctness bug and raises ``AssertionError``.
    """
    cascade_result = evaluate(source_code)
    if not cascade_result["accepted"]:
        return cascade_result

    kwargs: dict[str, Any] = {}
    if timeout_ms is not None:
        kwargs["timeout_ms"] = timeout_ms
    sandbox_result = sandbox.run_in_sandbox(source_code, input_value, **kwargs)

    status = sandbox_result["status"]
    assert status != "bad_source", (
        "cascade accepted source that sandbox rejected as bad_source; "
        f"cascade correctness bug. sandbox error: {sandbox_result.get('error')!r}"
    )

    duration_ms = float(sandbox_result["runtime_ms"])
    if status == "ok":
        return Result(
            accepted=True,
            stage="completed",
            reason=None,
            value=sandbox_result["output"],
            duration_ms=duration_ms,
            metadata={"sandbox_status": "ok"},
        )

    return Result(
        accepted=False,
        stage="sandbox",
        reason=f"sandbox_{status}: {sandbox_result.get('error')}",
        value=None,
        duration_ms=duration_ms,
        metadata={"sandbox_status": status},
    )
