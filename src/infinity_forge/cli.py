"""Command-line entry point for the forge loop.

Default path runs ``--iterations`` iterations against QwenGenerator, logging
to ``--log``. ``--sanity-check`` overrides the iteration count to exactly 3,
intended as the Colab smoke test before committing to a long run.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forge",
        description="Run the ∞ Forge generator loop.",
    )
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--log", type=Path, default=Path("forge_log.jsonl"))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Run exactly 3 iterations as a smoke test before a long run.",
    )
    parser.add_argument(
        "--multi-model",
        action="store_true",
        help="Round-robin across Qwen3-1.7B, Gemma-2-2B, and SmolLM2-1.7B instead of Qwen alone.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    n_iterations = 3 if args.sanity_check else args.iterations

    from infinity_forge.forge import run
    from infinity_forge.generator import (
        GemmaGenerator,
        MultiGenerator,
        PhiGenerator,
        QwenGenerator,
    )
    from infinity_forge.signatures import ACTIVE_SIGNATURES

    if args.multi_model:
        generator = MultiGenerator([
            ("qwen", QwenGenerator()),
            ("gemma", GemmaGenerator()),
            ("phi", PhiGenerator()),
        ])
    else:
        generator = QwenGenerator()

    run(
        generator=generator,
        log_path=args.log,
        n_iterations=n_iterations,
        active_signatures=ACTIVE_SIGNATURES,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
