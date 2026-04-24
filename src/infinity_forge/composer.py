"""Day 9: Level 2 composition engine.

Mechanically combines pairs of verified Level 1 atoms (A: X→Y composed
with B: Y→Z) into Level 2 atoms (f: X→Z), then runs each composition
through the same cascade + sandbox + Layer 6 novelty checks that gate
Level 1 output. No LLM is involved — composition is pure enumeration.

Layer 6 novelty for Level 2 is checked against the union of Level 1
and Level 2 fingerprint indexes, so compositions that happen to
duplicate atoms already found directly by the generator are rejected.
Canonical duplicates at the Level 1 or Level 2 layer are also rejected.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from infinity_forge.canonical import canonical_key, canonicalize
from infinity_forge.cascade import gate
from infinity_forge.inputs import sample_input
from infinity_forge.novelty import (
    append_to_index,
    compute_fingerprint,
    index_key,
    is_novel,
    load_index,
)


def _fingerprint_index_path(log_path: Path) -> Path:
    return log_path.with_name(log_path.stem + ".fingerprints.jsonl")


def extract_param_name(source: str) -> str:
    """Parse ``source`` and return the single positional argument's name."""
    tree = ast.parse(source)
    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("source must begin with a top-level FunctionDef")
    func = tree.body[0]
    positional = func.args.posonlyargs + func.args.args
    if len(positional) != 1:
        raise ValueError(
            f"expected exactly one positional argument (got {len(positional)})"
        )
    return positional[0].arg


def extract_body(source: str) -> str:
    """Return the function body as a string.

    Single-statement ``return <expr>`` bodies yield the expression only
    (suitable as a lambda body). Multi-statement or non-return bodies
    yield all statements joined by newlines — the composer falls back
    to a nested-``def`` form for these.
    """
    tree = ast.parse(source)
    func = tree.body[0]
    body = func.body
    if (
        len(body) == 1
        and isinstance(body[0], ast.Return)
        and body[0].value is not None
    ):
        return ast.unparse(body[0].value)
    return "\n".join(ast.unparse(stmt) for stmt in body)


def extract_body_statements(source: str) -> str:
    """Return the function body as statement(s), preserving ``return``.

    Unlike :func:`extract_body` (which strips ``return`` so the result
    can be used as a lambda body), this form is always valid as the
    body of a nested ``def``. A single ``return <expr>`` comes back as
    ``return <expr>``; multi-statement bodies are unparsed verbatim.
    """
    tree = ast.parse(source)
    func = tree.body[0]
    return "\n".join(ast.unparse(stmt) for stmt in func.body)


def _indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join((pad + line) if line else line for line in text.split("\n"))


def compose_source(
    atom_a_source: str,
    atom_a_param: str,
    atom_b_source: str,
    atom_b_param: str,
) -> str:
    """Mechanically compose two atoms into ``f(x) = B(A(x))``.

    Single-expression return bodies are inlined as lambdas; atoms with
    multi-statement bodies fall back to nested ``def`` wrappers so the
    composition is syntactically valid for any cascade-accepted atom.
    """
    body_a = extract_body(atom_a_source)
    body_b = extract_body(atom_b_source)

    lambda_safe = "\n" not in body_a and "\n" not in body_b
    if lambda_safe:
        return (
            f"def f({atom_a_param}):\n"
            f"    _a = lambda {atom_a_param}: {body_a}\n"
            f"    _b = lambda {atom_b_param}: {body_b}\n"
            f"    return _b(_a({atom_a_param}))\n"
        )

    nested_a = extract_body_statements(atom_a_source)
    nested_b = extract_body_statements(atom_b_source)
    return (
        f"def f({atom_a_param}):\n"
        f"    def _a({atom_a_param}):\n"
        f"{_indent(nested_a, 8)}\n"
        f"    def _b({atom_b_param}):\n"
        f"{_indent(nested_b, 8)}\n"
        f"    return _b(_a({atom_a_param}))\n"
    )


def find_composable_pairs(
    atoms: list[dict],
    signatures: list[tuple[str, str]],
) -> list[tuple[dict, dict, tuple[str, str]]]:
    """Return every ``(A, B, composed_sig)`` where ``A.output == B.input``.

    Self-composition (same source on both sides) is excluded, and the
    composed signature ``(A.input, B.output)`` must be in
    ``signatures`` — we only compose into cells we can fingerprint.
    """
    sig_set = {tuple(s) for s in signatures}
    by_output: dict[str, list[dict]] = {}
    by_input: dict[str, list[dict]] = {}
    for a in atoms:
        sig = tuple(a["signature"])
        by_output.setdefault(sig[1], []).append(a)
        by_input.setdefault(sig[0], []).append(a)

    pairs: list[tuple[dict, dict, tuple[str, str]]] = []
    for y, left_atoms in by_output.items():
        right_atoms = by_input.get(y, [])
        for a in left_atoms:
            for b in right_atoms:
                if a["source"] == b["source"]:
                    continue
                composed_sig = (
                    tuple(a["signature"])[0],
                    tuple(b["signature"])[1],
                )
                if composed_sig not in sig_set:
                    continue
                pairs.append((a, b, composed_sig))
    return pairs


def is_composition_ready(atom: dict) -> bool:
    """Check if an atom's fingerprint has any __raises__ slots.

    An atom that crashes on any of its 20 probes is not safe to
    compose with — its input domain is narrower than its signature
    promises. Such atoms stay in the Level 1 library (they're
    behaviorally valid and novel) but are excluded from Level 2
    composition to avoid cascading crashes.
    """
    fingerprint = atom.get("fingerprint", [])
    if not fingerprint:
        return False  # no fingerprint = can't verify = don't compose
    return not any(
        slot.startswith("__raises__") or slot == "__bad_output_type__"
        for slot in fingerprint
    )


def _load_accepted_atoms(log_path: Path) -> list[dict]:
    """Return ``[{"source", "signature", "fingerprint"}]`` for every accepted L1 atom."""
    if not log_path.exists():
        return []
    out: list[dict] = []
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sig = rec.get("signature")
            src = rec.get("extracted_source")
            gr = rec.get("gate_result")
            fp = rec.get("fingerprint")
            if (
                isinstance(sig, list)
                and len(sig) == 2
                and isinstance(src, str)
                and isinstance(gr, dict)
                and gr.get("accepted") is True
                and isinstance(fp, list)
            ):
                out.append(
                    {
                        "source": src,
                        "signature": (sig[0], sig[1]),
                        "fingerprint": list(fp),
                    }
                )
    return out


def _scan_level2_log(log_path: Path) -> tuple[int, list[tuple[tuple[str, str], str]]]:
    """Return ``(next_iteration, [(sig, source), ...])`` for the L2 log."""
    if not log_path.exists():
        return 0, []
    max_iter = -1
    accepted: list[tuple[tuple[str, str], str]] = []
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            it = rec.get("iteration")
            if isinstance(it, int) and it > max_iter:
                max_iter = it
            sig = rec.get("signature")
            src = rec.get("extracted_source")
            gr = rec.get("gate_result")
            if (
                isinstance(sig, list)
                and len(sig) == 2
                and isinstance(src, str)
                and isinstance(gr, dict)
                and gr.get("accepted") is True
            ):
                accepted.append(((sig[0], sig[1]), src))
    return max_iter + 1, accepted


def compose_run(
    level1_log_path: Path,
    output_log_path: Path,
    active_signatures: list[tuple[str, str]],
) -> None:
    """Run the Level 2 composition pass.

    Enumerates every type-compatible pair of accepted Level 1 atoms,
    composes each pair mechanically, and re-verifies the result through
    the same cascade, canonical-duplicate, and Layer 6 novelty checks
    that gate Level 1 atoms. Accepted compositions are appended to
    ``output_log_path`` with ``level=2`` and a ``components`` field.
    """
    level1_log_path = Path(level1_log_path)
    output_log_path = Path(output_log_path)
    output_log_path.parent.mkdir(parents=True, exist_ok=True)

    l1_fp_path = _fingerprint_index_path(level1_log_path)
    l2_fp_path = _fingerprint_index_path(output_log_path)
    combined_fp: dict[str, tuple[str, int]] = {}
    combined_fp.update(load_index(l1_fp_path))
    combined_fp.update(load_index(l2_fp_path))

    all_atoms = _load_accepted_atoms(level1_log_path)
    raises_count = sum(
        1
        for a in all_atoms
        if any(slot.startswith("__raises__") for slot in a.get("fingerprint", []))
    )
    bad_type_count = sum(
        1
        for a in all_atoms
        if any(slot == "__bad_output_type__" for slot in a.get("fingerprint", []))
    )
    atoms = [a for a in all_atoms if is_composition_ready(a)]
    filtered_out = len(all_atoms) - len(atoms)
    print(
        f"[composer] filtered {filtered_out} of {len(all_atoms)} L1 atoms as "
        f"not composition-ready ({raises_count} with __raises__, "
        f"{bad_type_count} with __bad_output_type__)",
        flush=True,
    )
    pairs = find_composable_pairs(atoms, active_signatures)

    canonical_seen: set[str] = set()
    for a in all_atoms:
        sig = tuple(a["signature"])
        canonical_seen.add(canonical_key(canonicalize(a["source"]), sig))

    next_iter, existing_l2 = _scan_level2_log(output_log_path)
    for sig, src in existing_l2:
        canonical_seen.add(canonical_key(canonicalize(src), sig))

    counts: dict[str, int] = {
        "pairs": len(pairs),
        "attempted": 0,
        "accepted": 0,
        "extract_fail": 0,
        "layer_1": 0,
        "layer_2": 0,
        "sandbox": 0,
        "canonical_dup": 0,
        "bad_type": 0,
        "fingerprint_dup": 0,
    }

    print(
        f"[composer] enumerating {len(pairs)} type-compatible pairs "
        f"from {len(atoms)} composition-ready L1 atoms",
        flush=True,
    )

    failures_path = output_log_path.with_suffix(".failures.jsonl")

    def _log_failure(
        fail_fh,
        atom_a: dict,
        atom_b: dict,
        composed_sig: tuple[str, str],
        composed_source: str,
        stage: str,
        reason: str,
    ) -> None:
        rec = {
            "atom_a": {
                "source": atom_a["source"],
                "signature": list(atom_a["signature"]),
            },
            "atom_b": {
                "source": atom_b["source"],
                "signature": list(atom_b["signature"]),
            },
            "composed_source": composed_source,
            "composed_signature": list(composed_sig),
            "rejection_stage": stage,
            "rejection_reason": reason,
        }
        fail_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fail_fh.flush()

    with output_log_path.open("a", encoding="utf-8") as fh, failures_path.open(
        "a", encoding="utf-8"
    ) as fail_fh:
        for pair_idx, (atom_a, atom_b, composed_sig) in enumerate(pairs):
            counts["attempted"] += 1
            try:
                param_a = extract_param_name(atom_a["source"])
                param_b = extract_param_name(atom_b["source"])
                source = compose_source(
                    atom_a["source"], param_a, atom_b["source"], param_b
                )
            except (SyntaxError, ValueError) as exc:
                counts["extract_fail"] += 1
                _log_failure(
                    fail_fh, atom_a, atom_b, composed_sig, "", "extract_fail", str(exc)
                )
                _maybe_progress(counts)
                continue

            input_value = sample_input(composed_sig[0], seed=pair_idx)
            gate_result = gate(source, input_value)
            if not gate_result["accepted"]:
                stage = gate_result["stage"]
                counts[stage] = counts.get(stage, 0) + 1
                _log_failure(
                    fail_fh,
                    atom_a,
                    atom_b,
                    composed_sig,
                    source,
                    stage,
                    str(gate_result.get("reason") or ""),
                )
                _maybe_progress(counts)
                continue

            ckey = canonical_key(canonicalize(source), composed_sig)
            if ckey in canonical_seen:
                counts["canonical_dup"] += 1
                _log_failure(
                    fail_fh,
                    atom_a,
                    atom_b,
                    composed_sig,
                    source,
                    "canonical",
                    "canonical duplicate of known atom",
                )
                _maybe_progress(counts)
                continue

            fingerprint = compute_fingerprint(source, composed_sig)
            if any(slot == "__bad_output_type__" for slot in fingerprint):
                counts["bad_type"] += 1
                _log_failure(
                    fail_fh,
                    atom_a,
                    atom_b,
                    composed_sig,
                    source,
                    "novelty",
                    "composition returned wrong output type on at least one probe",
                )
                _maybe_progress(counts)
                continue

            novel, _existing = is_novel(fingerprint, combined_fp, composed_sig)
            if not novel:
                counts["fingerprint_dup"] += 1
                _log_failure(
                    fail_fh,
                    atom_a,
                    atom_b,
                    composed_sig,
                    source,
                    "layer_6",
                    "fingerprint duplicate of existing atom",
                )
                _maybe_progress(counts)
                continue

            record: dict[str, Any] = {
                "iteration": next_iter,
                "level": 2,
                "signature": list(composed_sig),
                "extracted_source": source,
                "input_value": input_value,
                "gate_result": gate_result,
                "fingerprint": fingerprint,
                "components": [
                    {
                        "source": atom_a["source"],
                        "signature": list(atom_a["signature"]),
                    },
                    {
                        "source": atom_b["source"],
                        "signature": list(atom_b["signature"]),
                    },
                ],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()

            key = index_key(composed_sig, fingerprint)
            append_to_index(l2_fp_path, key, source, next_iter)
            combined_fp[key] = (source, next_iter)
            canonical_seen.add(ckey)
            counts["accepted"] += 1
            next_iter += 1
            _maybe_progress(counts)

    print(
        f"[composer] done: attempted={counts['attempted']} "
        f"accepted={counts['accepted']} "
        f"canonical_dup={counts['canonical_dup']} "
        f"bad_type={counts['bad_type']} "
        f"fingerprint_dup={counts['fingerprint_dup']} "
        f"layer_1={counts['layer_1']} "
        f"layer_2={counts['layer_2']} "
        f"sandbox={counts['sandbox']} "
        f"extract_fail={counts['extract_fail']}",
        flush=True,
    )


def _maybe_progress(counts: dict[str, int]) -> None:
    if counts["attempted"] % 100 == 0:
        print(
            f"[composer] {counts['attempted']}/{counts['pairs']} pairs: "
            f"accepted={counts['accepted']}",
            flush=True,
        )
