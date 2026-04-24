"""Prompt construction for the Day 3 generator.

`build_prompt` formats a terse instruction block that tells the model what
signature to target, what builtins are permitted, and what limits apply.
Optional few-shot examples (up to 3) are injected between instructions and
the final "write your function" line when provided.
"""

from infinity_forge.sandbox import _ALLOWED_BUILTIN_NAMES
from infinity_forge.signatures import describe_type

_BUILTIN_LIST: str = ", ".join(sorted(_ALLOWED_BUILTIN_NAMES))


def build_prompt(
    input_type: str,
    output_type: str,
    few_shot_atoms: list[str] | None = None,
) -> str:
    """Build the generator prompt for a given (input_type, output_type) pair."""
    input_desc = describe_type(input_type)
    output_desc = describe_type(output_type)

    lines: list[str] = [
        "Write a pure Python function named `f`.",
        f"It takes exactly one positional argument: {input_desc} ({input_type}).",
        f"It returns {output_desc} ({output_type}).",
        "The input is guaranteed to match the declared type. Do not validate or check the input — write the function body assuming correct input.",
        "Never call isinstance — it is not available and will cause rejection.",
        f"Allowed builtins only: {_BUILTIN_LIST}.",
        "No imports. No docstrings. No comments. No explanation.",
        "Maximum 30 lines. Cyclomatic complexity at most 10.",
    ]

    if few_shot_atoms:
        examples = few_shot_atoms[:3]
        lines.append("")
        lines.append("Examples:")
        for atom in examples:
            lines.append("```python")
            lines.append(atom.strip())
            lines.append("```")

    lines.append("")
    lines.append("Now write your function:")

    return "\n".join(lines)
