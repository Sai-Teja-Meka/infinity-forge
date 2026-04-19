"""One-shot helper to generate the frozen probe constants for probes.py.

Runs inputs.sample_input(input_type, seed) for seed in range(20) across all
8 input types and prints the result in a format that can be pasted directly
into src/infinity_forge/probes.py.

The probes are frozen forever once pasted — re-running this script with a
different inputs.py would produce different values and invalidate every
existing behavioral fingerprint. Keep this file as the provenance record
of how probes.py was originally generated.

Usage:
    python scripts/freeze_probes.py
"""
from __future__ import annotations

from infinity_forge.inputs import sample_input

INPUT_TYPES = [
    "int",
    "float",
    "bool",
    "str",
    "list[int]",
    "list[float]",
    "list[str]",
    "dict",
]


def main() -> None:
    print("PROBES: dict[str, list] = {")
    for input_type in INPUT_TYPES:
        values = [sample_input(input_type, seed) for seed in range(20)]
        print(f"    {input_type!r}: {values!r},")
    print("}")


if __name__ == "__main__":
    main()
