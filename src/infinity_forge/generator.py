"""Text generators for the Day 3 forge loop.

`Generator` is the abstract interface the forge depends on. `MockGenerator`
serves canned responses keyed by prompt hash and is the only generator used
in tests. `QwenGenerator` wraps a Qwen3-1.7B transformers pipeline; it must
never be instantiated from tests — loading the model takes minutes and
consumes gigabytes of RAM.

`extract_code` pulls the first Python function definition out of raw generator
output, handling three common shapes (```python fenced, ``` fenced, bare).
"""

from __future__ import annotations

import hashlib
import re
from typing import Any


class Generator:
    """Abstract text generator."""

    def generate(self, prompt: str, temperature: float) -> str:
        raise NotImplementedError


class MockGenerator(Generator):
    """Returns canned responses keyed by the hash of the incoming prompt.

    `responses` is a mapping from prompt-hash (hex SHA-256, first 16 chars)
    to the string to return. If a prompt isn't in the map, the `default`
    response is used. Used exclusively for tests.
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default: str = "",
        sequence: list[str] | None = None,
    ) -> None:
        self.responses: dict[str, str] = dict(responses or {})
        self.default: str = default
        self.sequence: list[str] | None = list(sequence) if sequence is not None else None
        self.sequence_index: int = 0
        self.calls: list[tuple[str, float]] = []

    @staticmethod
    def prompt_key(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

    def set(self, prompt: str, response: str) -> None:
        self.responses[self.prompt_key(prompt)] = response

    def generate(self, prompt: str, temperature: float) -> str:
        self.calls.append((prompt, temperature))
        if self.sequence is not None:
            if self.sequence_index < len(self.sequence):
                out = self.sequence[self.sequence_index]
                self.sequence_index += 1
                return out
            return self.default
        return self.responses.get(self.prompt_key(prompt), self.default)


_FENCED_PYTHON_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
_FENCED_ANY_RE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
_BARE_DEF_RE = re.compile(r"(?ms)^(def\s+f\s*\(.*)")


def _find_function_in(text: str) -> str | None:
    """Return the first `def f(...)` block in `text`, trimmed, or None."""
    match = _BARE_DEF_RE.search(text)
    if not match:
        return None
    start = match.start()
    lines = text[start:].splitlines()
    collected: list[str] = [lines[0]]
    for line in lines[1:]:
        if line.strip() == "":
            collected.append(line)
            continue
        # Continuation: either indented, or part of the def signature
        # (closing paren / return annotation on its own line).
        if line.startswith((" ", "\t")):
            collected.append(line)
        else:
            break
    # Strip trailing blank lines
    while collected and collected[-1].strip() == "":
        collected.pop()
    body = "\n".join(collected)
    return body if body else None


def extract_code(raw: str) -> str | None:
    """Extract the first Python `def f(...)` function from a raw generation.

    Checks, in order: a ```python fenced block, a ``` fenced block, then a
    bare `def f(...)` in the text. Returns the function source as a string,
    or None if nothing recognizable was produced.
    """
    if not raw:
        return None

    for pattern in (_FENCED_PYTHON_RE, _FENCED_ANY_RE):
        for m in pattern.finditer(raw):
            body = m.group(1)
            fn = _find_function_in(body)
            if fn is not None:
                return fn

    return _find_function_in(raw)


class QwenGenerator(Generator):
    """Loads Qwen3-1.7B via transformers pipeline on first generate() call.

    Lazy-loads so importing this module (including during test collection)
    does not touch the model. Device selection is automatic: CUDA if
    available, else CPU. Loading prints the selected device so Colab runs
    confirm GPU usage visually.

    Never instantiate from tests.
    """

    MODEL_ID: str = "Qwen/Qwen3-1.7B"

    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._device: str | None = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        import torch  # type: ignore
        from transformers import pipeline  # type: ignore

        if torch.cuda.is_available():
            device = "cuda:0"
            torch_dtype = torch.float16
        else:
            device = "cpu"
            torch_dtype = torch.float32

        print(f"QwenGenerator: using device={device}")

        self._pipeline = pipeline(
            task="text-generation",
            model=self.MODEL_ID,
            torch_dtype=torch_dtype,
            device=device,
        )
        self._device = device

    def generate(self, prompt: str, temperature: float) -> str:
        self._load()
        assert self._pipeline is not None
        out = self._pipeline(
            prompt,
            max_new_tokens=400,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            return_full_text=False,
        )
        return out[0]["generated_text"]
