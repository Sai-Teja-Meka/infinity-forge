"""Generator loop with Day 4 additions: seeds, Layer 6, fingerprint-diverse few-shot.

Prompt a text generator for candidate Python atoms, run each candidate
through the cascade ``gate``, run accepted atoms through Layer 6 (behavioral
fingerprinting), and append every outcome as a line of JSON to ``log_path``.

Day 4 additions:
  * On fresh run (``log_path`` does not exist), inject :data:`seeds.SEED_ATOMS`
    at iterations ``-1, -2, -3, ...`` before the first generator iteration.
    Each seed computes its own fingerprint and appends to the fingerprint
    index side file.
  * After :func:`cascade.gate` accepts, compute the fingerprint and run
    novelty/type checks. Rejections at this stage override ``gate_result``
    with ``stage="layer_6"`` (type mismatch) or ``stage="novelty"`` (duplicate).
  * Few-shot selection pulls up to 3 accepted atoms per signature with
    **distinct fingerprints**, newest first, so the example pool cannot
    degenerate into three copies of the same atom.

The loop remains resumable (picks up from the max iteration already in the
file — seeds have negative iterations, so they are always "before" the
generator iterations and do not re-inject on resume).
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from infinity_forge.cascade import Result, gate
from infinity_forge.generator import Generator, extract_code
from infinity_forge.inputs import sample_input
from infinity_forge.novelty import (
    Fingerprint,
    append_to_index,
    compute_fingerprint,
    index_key,
    is_novel,
    load_index,
)
from infinity_forge.probes import EXPECTED_OUTPUT_TYPES
from infinity_forge.prompts import build_prompt
from infinity_forge.seeds import SEED_ATOMS
from infinity_forge.signatures import ACTIVE_SIGNATURES

_TEMPERATURES: tuple[float, ...] = (0.7, 0.9, 1.1)
_FEW_SHOT_THRESHOLD: int = 10
_STATUS_INTERVAL: int = 25
_PER_SIG_INTERVAL: int = 100

_CSV_HEADER: tuple[str, ...] = (
    "iteration",
    "signature_in",
    "signature_out",
    "stage",
    "accepted",
    "cum_accepted",
    "cum_dup",
    "cum_bad",
    "cum_l1",
    "cum_l2",
    "cum_sandbox",
)


def _empty_per_sig_counts() -> dict[str, int]:
    return {"total": 0, "novel": 0, "dup": 0, "bad": 0, "L2": 0, "sbx": 0}


@dataclass
class IterationResult:
    iteration: int
    signature: tuple[str, str]
    temperature: float
    raw_llm_output: str
    extracted_source: str | None
    input_value: Any
    gate_result: Result | None
    fingerprint: Fingerprint | None
    timestamp: str

    def to_json_line(self) -> str:
        d = asdict(self)
        d["signature"] = list(self.signature)
        return json.dumps(d, ensure_ascii=False)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fingerprint_index_path(log_path: Path) -> Path:
    """Index side-file path derived from the log path.

    For ``forge.jsonl`` this yields ``forge.fingerprints.jsonl`` alongside.
    """
    return log_path.with_name(log_path.stem + ".fingerprints.jsonl")


def _csv_metrics_path(log_path: Path) -> Path:
    """CSV metrics side-file path. For ``forge.jsonl`` → ``forge.csv``."""
    return log_path.with_suffix(".csv")


def _stage_from_record(
    extracted_source: str | None,
    gate_result: Result | None,
) -> str:
    """Terminal stage for an iteration, for CSV ``stage`` column."""
    if extracted_source is None:
        return "extract_fail"
    if gate_result is None:
        return "extract_fail"
    if gate_result["accepted"]:
        return "completed"
    return gate_result["stage"]


def _first_bad_type_probe_index(fingerprint: Fingerprint) -> int | None:
    for i, slot in enumerate(fingerprint):
        if slot == "__bad_output_type__":
            return i
    return None


def _pick_few_shot(
    pool: list[tuple[str, Fingerprint]],
    k: int = 3,
) -> list[str]:
    """Pick up to k atoms from ``pool`` with distinct fingerprints, newest first."""
    seen: set[str] = set()
    picked: list[str] = []
    for source, fp in reversed(pool):
        fp_key = json.dumps(fp, separators=(",", ":"))
        if fp_key in seen:
            continue
        seen.add(fp_key)
        picked.append(source)
        if len(picked) >= k:
            break
    return picked


def _read_existing(
    log_path: Path,
) -> tuple[int, dict[tuple[str, str], list[tuple[str, Fingerprint]]]]:
    """Return (next_iteration_index, accepted_by_sig) from log_path.

    ``accepted_by_sig`` is ``{sig: [(source, fingerprint), ...]}`` in insertion
    order (oldest first). Entries missing a fingerprint are skipped from the
    few-shot pool — they remain in the log but cannot participate in the
    distinct-fingerprint few-shot selection.

    Seed entries (negative iteration) ARE included in ``accepted_by_sig`` so
    few-shot can pull from them. They do NOT influence ``next_iteration_index``,
    which considers non-negative iterations only.
    """
    if not log_path.exists():
        return 0, {}

    max_iter = -1
    accepted: dict[tuple[str, str], list[tuple[str, Fingerprint]]] = {}
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
            if isinstance(it, int) and it >= 0 and it > max_iter:
                max_iter = it
            sig = rec.get("signature")
            gr = rec.get("gate_result")
            src = rec.get("extracted_source")
            fp = rec.get("fingerprint")
            if (
                isinstance(sig, list)
                and len(sig) == 2
                and isinstance(gr, dict)
                and gr.get("accepted") is True
                and isinstance(src, str)
                and isinstance(fp, list)
            ):
                key = (sig[0], sig[1])
                accepted.setdefault(key, []).append((src, list(fp)))
    return max_iter + 1, accepted


def _inject_seeds(
    log_fh,
    log_path: Path,
    fp_index_path: Path,
    fp_index: dict[str, tuple[str, int]],
    accepted_by_sig: dict[tuple[str, str], list[tuple[str, Fingerprint]]],
) -> None:
    """Write each seed atom to the log + fingerprint index. Iterations are -1, -2, ..."""
    for i, seed in enumerate(SEED_ATOMS):
        iteration = -(i + 1)
        sig = tuple(seed["signature"])
        source = seed["source"]
        input_value = sample_input(sig[0], seed=0)
        gate_result = gate(source, input_value)
        fingerprint = compute_fingerprint(source, sig)

        record = IterationResult(
            iteration=iteration,
            signature=sig,
            temperature=0.0,
            raw_llm_output=seed.get("note", ""),
            extracted_source=source,
            input_value=input_value,
            gate_result=gate_result,
            fingerprint=fingerprint,
            timestamp=_now_iso(),
        )
        log_fh.write(record.to_json_line())
        log_fh.write("\n")
        log_fh.flush()

        key = index_key(sig, fingerprint)
        append_to_index(fp_index_path, key, source, iteration)
        fp_index[key] = (source, iteration)
        accepted_by_sig.setdefault(sig, []).append((source, fingerprint))


def _print_status(
    start: int,
    current: int,
    totals: dict[str, int],
) -> None:
    processed = current - start + 1
    with_gate = totals["gate_ran"]
    accepts = totals["accepted"]
    rate = (accepts / with_gate * 100.0) if with_gate else 0.0
    print(
        f"[forge] iter {start}..{current} "
        f"({processed} done) | "
        f"accept {accepts}/{with_gate} ({rate:.1f}%) | "
        f"extract_fail={totals['extract_fail']} "
        f"layer1={totals['reject_layer_1']} "
        f"layer2={totals['reject_layer_2']} "
        f"sandbox={totals['reject_sandbox']} "
        f"l6type={totals['reject_layer_6']} "
        f"novelty={totals['reject_novelty']}",
        flush=True,
    )


def _print_per_signature(
    current: int,
    sigs: list[tuple[str, str]],
    per_sig: dict[tuple[str, str], dict[str, int]],
) -> None:
    label_width = max(len(f"{s[0]} -> {s[1]}") for s in sigs)
    print(f"[forge] per-signature at iter {current + 1}:", flush=True)
    for sig in sigs:
        c = per_sig.get(sig, _empty_per_sig_counts())
        label = f"{sig[0]} -> {sig[1]}".ljust(label_width)
        print(
            f"  {label} : "
            f"total={c['total']}  "
            f"novel={c['novel']}  "
            f"dup={c['dup']}  "
            f"bad={c['bad']}  "
            f"L2={c['L2']}  "
            f"sbx={c['sbx']}",
            flush=True,
        )


def run(
    generator: Generator,
    log_path: Path,
    n_iterations: int,
    active_signatures: list[tuple[str, str]] | None = None,
    resume: bool = True,
) -> None:
    """Run the forge for ``n_iterations`` iterations, appending to ``log_path``.

    On fresh run (``log_path`` does not yet exist), seeds from
    :data:`seeds.SEED_ATOMS` are injected with negative iteration numbers
    before the first generator iteration. Each accepted atom goes through
    the full Day 4 pipeline: cascade → sandbox → Layer 6 (type + novelty).
    """
    sigs = list(active_signatures) if active_signatures is not None else list(ACTIVE_SIGNATURES)
    if not sigs:
        raise ValueError("active_signatures must be non-empty")

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fp_index_path = _fingerprint_index_path(log_path)
    csv_path = _csv_metrics_path(log_path)

    fresh_run = not log_path.exists()
    csv_fresh = not csv_path.exists()

    if resume:
        start, accepted_by_sig = _read_existing(log_path)
    else:
        start, accepted_by_sig = 0, {}

    fp_index = load_index(fp_index_path)

    if start > 0:
        m = sum(len(v) for v in accepted_by_sig.values())
        k = len(accepted_by_sig)
        f = len(fp_index)
        print(
            f"[forge] resuming from iteration {start}, "
            f"found {m} accepted atoms across {k} signatures, "
            f"fingerprint index has {f} entries",
            flush=True,
        )

    totals = {
        "gate_ran": 0,
        "accepted": 0,
        "extract_fail": 0,
        "reject_layer_1": 0,
        "reject_layer_2": 0,
        "reject_sandbox": 0,
        "reject_layer_6": 0,
        "reject_novelty": 0,
    }

    per_sig: dict[tuple[str, str], dict[str, int]] = {
        s: _empty_per_sig_counts() for s in sigs
    }

    end = start + n_iterations
    # CSV cumulatives are running totals from THIS process's start. On resume,
    # any pre-existing CSV rows from prior runs are preserved as-is, and the
    # new rows' cum_* columns start at the current-run deltas (beginning at the
    # first completed iteration of this invocation). Day 5.5 analysis scripts
    # must detect and stitch these discontinuities by iteration boundary.
    with log_path.open("a", encoding="utf-8") as fh, csv_path.open(
        "a", encoding="utf-8", newline=""
    ) as csv_fh:
        csv_writer = csv.writer(csv_fh)
        if csv_fresh:
            csv_writer.writerow(_CSV_HEADER)
            csv_fh.flush()

        if fresh_run:
            _inject_seeds(fh, log_path, fp_index_path, fp_index, accepted_by_sig)

        for i in range(start, end):
            sig = sigs[i % len(sigs)]
            temperature = _TEMPERATURES[i % len(_TEMPERATURES)]

            pool = accepted_by_sig.get(sig, [])
            few_shot = (
                _pick_few_shot(pool, k=3)
                if len(pool) >= _FEW_SHOT_THRESHOLD
                else None
            )

            prompt = build_prompt(sig[0], sig[1], few_shot_atoms=few_shot)
            raw = generator.generate(prompt, temperature)
            source = extract_code(raw)

            input_value: Any = None
            gate_result: Result | None = None
            fingerprint: Fingerprint | None = None

            sig_counts = per_sig.setdefault(sig, _empty_per_sig_counts())
            sig_counts["total"] += 1

            if source is None:
                totals["extract_fail"] += 1
            else:
                input_value = sample_input(sig[0], seed=i)
                gate_result = gate(source, input_value)
                totals["gate_ran"] += 1

                if gate_result["accepted"]:
                    fingerprint = compute_fingerprint(source, sig)
                    bad_idx = _first_bad_type_probe_index(fingerprint)
                    if bad_idx is not None:
                        expected = EXPECTED_OUTPUT_TYPES[sig[1]].__name__
                        gate_result = Result(
                            accepted=False,
                            stage="layer_6",
                            reason=(
                                f"layer_6: output type mismatch on probe[{bad_idx}]: "
                                f"expected {expected}"
                            ),
                            value=gate_result.get("value"),
                            duration_ms=gate_result.get("duration_ms"),
                            metadata={**gate_result.get("metadata", {}), "layer_6_reject": "type"},
                        )
                        totals["reject_layer_6"] += 1
                        sig_counts["bad"] += 1
                    else:
                        novel, existing_source = is_novel(fingerprint, fp_index, sig)
                        if not novel:
                            key = index_key(sig, fingerprint)
                            existing_iter = fp_index[key][1]
                            gate_result = Result(
                                accepted=False,
                                stage="novelty",
                                reason=f"layer_6: duplicate of atom from iteration {existing_iter}",
                                value=gate_result.get("value"),
                                duration_ms=gate_result.get("duration_ms"),
                                metadata={
                                    **gate_result.get("metadata", {}),
                                    "layer_6_reject": "duplicate",
                                    "duplicate_of_iteration": existing_iter,
                                },
                            )
                            totals["reject_novelty"] += 1
                            sig_counts["dup"] += 1
                        else:
                            totals["accepted"] += 1
                            sig_counts["novel"] += 1
                            key = index_key(sig, fingerprint)
                            append_to_index(fp_index_path, key, source, i)
                            fp_index[key] = (source, i)
                            accepted_by_sig.setdefault(sig, []).append(
                                (source, fingerprint)
                            )
                else:
                    stage = gate_result["stage"]
                    if stage == "layer_1":
                        totals["reject_layer_1"] += 1
                    elif stage == "layer_2":
                        totals["reject_layer_2"] += 1
                        sig_counts["L2"] += 1
                    elif stage == "sandbox":
                        totals["reject_sandbox"] += 1
                        sig_counts["sbx"] += 1

            record = IterationResult(
                iteration=i,
                signature=sig,
                temperature=temperature,
                raw_llm_output=raw,
                extracted_source=source,
                input_value=input_value,
                gate_result=gate_result,
                fingerprint=fingerprint,
                timestamp=_now_iso(),
            )
            fh.write(record.to_json_line())
            fh.write("\n")
            fh.flush()

            stage = _stage_from_record(source, gate_result)
            accepted_flag = 1 if (gate_result is not None and gate_result["accepted"]) else 0
            csv_writer.writerow(
                (
                    i,
                    sig[0],
                    sig[1],
                    stage,
                    accepted_flag,
                    totals["accepted"],
                    totals["reject_novelty"],
                    totals["reject_layer_6"],
                    totals["reject_layer_1"],
                    totals["reject_layer_2"],
                    totals["reject_sandbox"],
                )
            )
            csv_fh.flush()

            if (i + 1) % _STATUS_INTERVAL == 0:
                _print_status(start, i, totals)

            if (i + 1) % _PER_SIG_INTERVAL == 0:
                _print_per_signature(i, sigs, per_sig)

        print(
            f"[forge] completed {n_iterations} iterations: "
            f"{totals['accepted']} accepted, "
            f"{totals['reject_novelty']} duplicates, "
            f"{totals['reject_layer_6']} bad-type, "
            f"{totals['reject_layer_1']} layer_1, "
            f"{totals['reject_layer_2']} layer_2, "
            f"{totals['reject_sandbox']} sandbox, "
            f"{totals['extract_fail']} extract_fail",
            flush=True,
        )
