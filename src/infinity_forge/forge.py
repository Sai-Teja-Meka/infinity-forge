"""Day 3 generator loop.

Prompt a text generator for candidate Python atoms, run each candidate
through the cascade ``gate``, and append every outcome as a line of JSON to
``log_path``. The loop is resumable (picks up from the max iteration already
in the file) and observable (prints a status line every 25 iterations).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from infinity_forge.cascade import Result, gate
from infinity_forge.generator import Generator, extract_code
from infinity_forge.inputs import sample_input
from infinity_forge.prompts import build_prompt
from infinity_forge.signatures import ACTIVE_SIGNATURES

_TEMPERATURES: tuple[float, ...] = (0.7, 0.9, 1.1)
_FEW_SHOT_THRESHOLD: int = 10
_STATUS_INTERVAL: int = 25


@dataclass
class IterationResult:
    iteration: int
    signature: tuple[str, str]
    temperature: float
    raw_llm_output: str
    extracted_source: str | None
    input_value: Any
    gate_result: Result | None
    timestamp: str

    def to_json_line(self) -> str:
        d = asdict(self)
        d["signature"] = list(self.signature)
        return json.dumps(d, ensure_ascii=False)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_existing(log_path: Path) -> tuple[int, dict[tuple[str, str], list[str]]]:
    """Return (next_iteration_index, accepted_atoms_by_signature) from log_path.

    If the file does not exist or is empty, returns (0, {}).
    Lines that fail to parse are skipped.
    """
    if not log_path.exists():
        return 0, {}

    max_iter = -1
    accepted: dict[tuple[str, str], list[str]] = {}
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
            gr = rec.get("gate_result")
            src = rec.get("extracted_source")
            if (
                isinstance(sig, list)
                and len(sig) == 2
                and isinstance(gr, dict)
                and gr.get("accepted") is True
                and isinstance(src, str)
            ):
                key = (sig[0], sig[1])
                accepted.setdefault(key, []).append(src)
    return max_iter + 1, accepted


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
        f"sandbox={totals['reject_sandbox']}",
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

    See module docstring for semantics. The function prints status every
    25 iterations and re-raises any unhandled exception after flushing
    whatever was written so far.
    """
    sigs = list(active_signatures) if active_signatures is not None else list(ACTIVE_SIGNATURES)
    if not sigs:
        raise ValueError("active_signatures must be non-empty")

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if resume:
        start, accepted_by_sig = _read_existing(log_path)
    else:
        start, accepted_by_sig = 0, {}

    totals = {
        "gate_ran": 0,
        "accepted": 0,
        "extract_fail": 0,
        "reject_layer_1": 0,
        "reject_layer_2": 0,
        "reject_sandbox": 0,
    }

    end = start + n_iterations
    with log_path.open("a", encoding="utf-8") as fh:
        for i in range(start, end):
            sig = sigs[i % len(sigs)]
            temperature = _TEMPERATURES[i % len(_TEMPERATURES)]

            few_shot_pool = accepted_by_sig.get(sig, [])
            few_shot = few_shot_pool[:3] if len(few_shot_pool) >= _FEW_SHOT_THRESHOLD else None

            prompt = build_prompt(sig[0], sig[1], few_shot_atoms=few_shot)
            raw = generator.generate(prompt, temperature)
            source = extract_code(raw)

            input_value: Any = None
            gate_result: Result | None = None
            if source is None:
                totals["extract_fail"] += 1
            else:
                input_value = sample_input(sig[0], seed=i)
                gate_result = gate(source, input_value)
                totals["gate_ran"] += 1
                if gate_result["accepted"]:
                    totals["accepted"] += 1
                    accepted_by_sig.setdefault(sig, []).append(source)
                else:
                    stage = gate_result["stage"]
                    if stage == "layer_1":
                        totals["reject_layer_1"] += 1
                    elif stage == "layer_2":
                        totals["reject_layer_2"] += 1
                    elif stage == "sandbox":
                        totals["reject_sandbox"] += 1

            record = IterationResult(
                iteration=i,
                signature=sig,
                temperature=temperature,
                raw_llm_output=raw,
                extracted_source=source,
                input_value=input_value,
                gate_result=gate_result,
                timestamp=_now_iso(),
            )
            fh.write(record.to_json_line())
            fh.write("\n")
            fh.flush()

            if (i + 1) % _STATUS_INTERVAL == 0:
                _print_status(start, i, totals)
