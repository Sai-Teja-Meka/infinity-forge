"""Day 9: Level 2 composition engine.

Mechanically combines pairs of verified Level 1 atoms (A: X→Y composed
with B: Y→Z) into Level 2 atoms (f: X→Z), then runs each composition
through the same cascade + sandbox + Layer 6 novelty checks that gate
Level 1 output. No LLM is involved — composition is pure enumeration.

Layer 6 novelty for Level 2 is checked against the union of Level 1
and Level 2 fingerprint indexes, so compositions that happen to
duplicate atoms already found directly by the generator are rejected.
Canonical duplicates at the Level 1 or Level 2 layer are also rejected.

Day 11 extends the same machinery to Level 3 by composing a Level 2
atom with a Level 1 atom in either direction; verification, dedupe,
and logging share a single driver with Level 2.
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
    """Return ``[{"source", "signature", "fingerprint"}]`` for every accepted atom.

    Works for any composition log (L1, L2, L3) since they share the
    ``extracted_source`` / ``signature`` / ``fingerprint`` / ``gate_result``
    schema.
    """
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


def _scan_compose_log(log_path: Path) -> tuple[int, list[tuple[tuple[str, str], str]]]:
    """Return ``(next_iteration, [(sig, source), ...])`` for any composition log."""
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


def _filter_composition_ready(
    atoms: list[dict], layer_label: str, log_label: str = "composer"
) -> list[dict]:
    """Filter atoms for composition-readiness and print a diagnostic line."""
    raises_count = sum(
        1
        for a in atoms
        if any(slot.startswith("__raises__") for slot in a.get("fingerprint", []))
    )
    bad_type_count = sum(
        1
        for a in atoms
        if any(slot == "__bad_output_type__" for slot in a.get("fingerprint", []))
    )
    ready = [a for a in atoms if is_composition_ready(a)]
    filtered_out = len(atoms) - len(ready)
    print(
        f"[{log_label}] filtered {filtered_out} of {len(atoms)} {layer_label} atoms as "
        f"not composition-ready ({raises_count} with __raises__, "
        f"{bad_type_count} with __bad_output_type__)",
        flush=True,
    )
    return ready


def _seed_canonical(atoms: list[dict]) -> set[str]:
    out: set[str] = set()
    for a in atoms:
        sig = tuple(a["signature"])
        out.add(canonical_key(canonicalize(a["source"]), sig))
    return out


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


_STAGE_TO_BUCKET = {
    "extract_fail": "extract_fail",
    "layer_1": "layer_1",
    "layer_2": "layer_2",
    "sandbox": "sandbox",
    "canonical": "canonical_dup",
    "novelty": "bad_type",
    "layer_6": "fingerprint_dup",
}


def _bump_counter(counts: dict[str, int], stage: str) -> None:
    bucket = _STAGE_TO_BUCKET.get(stage, stage)
    counts[bucket] = counts.get(bucket, 0) + 1


def _verify_pair(
    atom_a: dict,
    atom_b: dict,
    composed_sig: tuple[str, str],
    pair_idx: int,
    *,
    canonical_seen: set[str],
    combined_fp: dict[str, tuple[str, int]],
) -> dict:
    """Run the verification pipeline for a single pair.

    Returns ``{"status": "accepted", ...}`` with source / fingerprint /
    input_value / gate_result / ckey, or ``{"status": "rejected", ...}``
    with source / stage / reason. Pure over the supplied dedupe state —
    no I/O.
    """
    try:
        param_a = extract_param_name(atom_a["source"])
        param_b = extract_param_name(atom_b["source"])
        source = compose_source(
            atom_a["source"], param_a, atom_b["source"], param_b
        )
    except (SyntaxError, ValueError) as exc:
        return {
            "status": "rejected",
            "source": "",
            "stage": "extract_fail",
            "reason": str(exc),
        }

    input_value = sample_input(composed_sig[0], seed=pair_idx)
    gate_result = gate(source, input_value)
    if not gate_result["accepted"]:
        return {
            "status": "rejected",
            "source": source,
            "stage": gate_result["stage"],
            "reason": str(gate_result.get("reason") or ""),
        }

    ckey = canonical_key(canonicalize(source), composed_sig)
    if ckey in canonical_seen:
        return {
            "status": "rejected",
            "source": source,
            "stage": "canonical",
            "reason": "canonical duplicate of known atom",
        }

    fingerprint = compute_fingerprint(source, composed_sig)
    if any(slot == "__bad_output_type__" for slot in fingerprint):
        return {
            "status": "rejected",
            "source": source,
            "stage": "novelty",
            "reason": "composition returned wrong output type on at least one probe",
        }

    novel, _existing = is_novel(fingerprint, combined_fp, composed_sig)
    if not novel:
        return {
            "status": "rejected",
            "source": source,
            "stage": "layer_6",
            "reason": "fingerprint duplicate of existing atom",
        }

    return {
        "status": "accepted",
        "source": source,
        "fingerprint": fingerprint,
        "input_value": input_value,
        "gate_result": gate_result,
        "ckey": ckey,
    }


def _run_composition_pass(
    *,
    pairs: list[tuple[dict, dict, tuple[str, str]]],
    level: int,
    output_log_path: Path,
    own_fp_path: Path,
    dedupe_fp_paths: list[Path],
    canonical_seen: set[str],
    next_iter: int,
    label: str,
) -> None:
    """Generic driver: verify each pair, write accept/failure records, log."""
    combined_fp: dict[str, tuple[str, int]] = {}
    for fp_path in dedupe_fp_paths:
        combined_fp.update(load_index(fp_path))

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

    failures_path = output_log_path.with_suffix(".failures.jsonl")
    with output_log_path.open("a", encoding="utf-8") as fh, failures_path.open(
        "a", encoding="utf-8"
    ) as fail_fh:
        for pair_idx, (atom_a, atom_b, composed_sig) in enumerate(pairs):
            counts["attempted"] += 1
            result = _verify_pair(
                atom_a,
                atom_b,
                composed_sig,
                pair_idx,
                canonical_seen=canonical_seen,
                combined_fp=combined_fp,
            )
            if result["status"] == "rejected":
                _bump_counter(counts, result["stage"])
                _log_failure(
                    fail_fh,
                    atom_a,
                    atom_b,
                    composed_sig,
                    result["source"],
                    result["stage"],
                    result["reason"],
                )
                _maybe_progress(counts, label)
                continue

            record: dict[str, Any] = {
                "iteration": next_iter,
                "level": level,
                "signature": list(composed_sig),
                "extracted_source": result["source"],
                "input_value": result["input_value"],
                "gate_result": result["gate_result"],
                "fingerprint": result["fingerprint"],
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

            key = index_key(composed_sig, result["fingerprint"])
            append_to_index(own_fp_path, key, result["source"], next_iter)
            combined_fp[key] = (result["source"], next_iter)
            canonical_seen.add(result["ckey"])
            counts["accepted"] += 1
            next_iter += 1
            _maybe_progress(counts, label)

    print(
        f"[{label}] done: attempted={counts['attempted']} "
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

    all_atoms = _load_accepted_atoms(level1_log_path)
    atoms = _filter_composition_ready(all_atoms, "L1", log_label="composer")
    pairs = find_composable_pairs(atoms, active_signatures)

    canonical_seen = _seed_canonical(all_atoms)
    next_iter, existing_l2 = _scan_compose_log(output_log_path)
    for sig, src in existing_l2:
        canonical_seen.add(canonical_key(canonicalize(src), sig))

    print(
        f"[composer] enumerating {len(pairs)} type-compatible pairs "
        f"from {len(atoms)} composition-ready L1 atoms",
        flush=True,
    )

    _run_composition_pass(
        pairs=pairs,
        level=2,
        output_log_path=output_log_path,
        own_fp_path=l2_fp_path,
        dedupe_fp_paths=[l1_fp_path, l2_fp_path],
        canonical_seen=canonical_seen,
        next_iter=next_iter,
        label="composer",
    )


def _find_l3_pairs(
    l1_atoms: list[dict],
    l2_atoms: list[dict],
    active_signatures: list[tuple[str, str]],
) -> list[tuple[dict, dict, tuple[str, str]]]:
    """Enumerate (L2-inner, L1-outer) and (L1-inner, L2-outer) pairs.

    Self-composition (same source on both sides) is excluded, mirroring
    the L1×L1 enumerator.
    """
    sig_set = {tuple(s) for s in active_signatures}
    pairs: list[tuple[dict, dict, tuple[str, str]]] = []

    for inner in l2_atoms:
        for outer in l1_atoms:
            if inner["source"] == outer["source"]:
                continue
            if inner["signature"][1] != outer["signature"][0]:
                continue
            csig = (inner["signature"][0], outer["signature"][1])
            if csig in sig_set:
                pairs.append((inner, outer, csig))

    for inner in l1_atoms:
        for outer in l2_atoms:
            if inner["source"] == outer["source"]:
                continue
            if inner["signature"][1] != outer["signature"][0]:
                continue
            csig = (inner["signature"][0], outer["signature"][1])
            if csig in sig_set:
                pairs.append((inner, outer, csig))

    return pairs


def compose_run_l3(
    level1_log_path: Path,
    level2_log_path: Path,
    output_log_path: Path,
    active_signatures: list[tuple[str, str]],
) -> None:
    """Run the Level 3 composition pass.

    Enumerates every type-compatible (L2, L1) pair in both directions
    (L2 inner → L1 outer, and L1 inner → L2 outer), composes each pair
    mechanically, and re-verifies through the same pipeline as Level 2.
    Dedupe combines L1, L2, and L3-so-far fingerprint indexes; canonical
    duplicates against L1, L2, and previously accepted L3 atoms in this
    run are rejected.
    """
    level1_log_path = Path(level1_log_path)
    level2_log_path = Path(level2_log_path)
    output_log_path = Path(output_log_path)
    output_log_path.parent.mkdir(parents=True, exist_ok=True)

    l1_fp_path = _fingerprint_index_path(level1_log_path)
    l2_fp_path = _fingerprint_index_path(level2_log_path)
    l3_fp_path = _fingerprint_index_path(output_log_path)

    all_l1 = _load_accepted_atoms(level1_log_path)
    all_l2 = _load_accepted_atoms(level2_log_path)
    l1 = _filter_composition_ready(all_l1, "L1", log_label="composer-l3")
    l2 = _filter_composition_ready(all_l2, "L2", log_label="composer-l3")

    pairs = _find_l3_pairs(l1, l2, active_signatures)

    canonical_seen = _seed_canonical(all_l1) | _seed_canonical(all_l2)
    next_iter, existing_l3 = _scan_compose_log(output_log_path)
    for sig, src in existing_l3:
        canonical_seen.add(canonical_key(canonicalize(src), sig))

    print(
        f"[composer-l3] enumerating {len(pairs)} type-compatible pairs "
        f"from {len(l1)} L1 + {len(l2)} L2 composition-ready atoms",
        flush=True,
    )

    _run_composition_pass(
        pairs=pairs,
        level=3,
        output_log_path=output_log_path,
        own_fp_path=l3_fp_path,
        dedupe_fp_paths=[l1_fp_path, l2_fp_path, l3_fp_path],
        canonical_seen=canonical_seen,
        next_iter=next_iter,
        label="composer-l3",
    )


def _maybe_progress(counts: dict[str, int], label: str = "composer") -> None:
    if counts["attempted"] % 100 == 0:
        print(
            f"[{label}] {counts['attempted']}/{counts['pairs']} pairs: "
            f"accepted={counts['accepted']}",
            flush=True,
        )
