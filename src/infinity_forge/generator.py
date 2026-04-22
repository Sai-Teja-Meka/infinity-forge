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
from abc import ABC, abstractmethod


class Generator(ABC):
    @abstractmethod
    def generate(self, prompt: str, temperature: float) -> str:
        ...


class MockGenerator(Generator):
    """Canned responses keyed by SHA-256 prompt hash. Used in all tests."""

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default: str = "",
        sequence: list[str] | None = None,
    ):
        self._responses = responses or {}
        self._default = default
        self._sequence = list(sequence) if sequence is not None else None
        self._sequence_idx = 0
        self.calls: list[tuple[str, float]] = []

    @staticmethod
    def prompt_key(prompt: str) -> str:
        """First 16 hex chars of SHA-256(prompt), used as the dict key."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]

    def set(self, prompt: str, response: str) -> None:
        """Register a canned response for the given prompt."""
        self._responses[self.prompt_key(prompt)] = response

    def generate(self, prompt: str, temperature: float) -> str:
        self.calls.append((prompt, temperature))
        if self._sequence is not None:
            if self._sequence_idx >= len(self._sequence):
                return self._default
            out = self._sequence[self._sequence_idx]
            self._sequence_idx += 1
            return out
        key = self.prompt_key(prompt)
        return self._responses.get(key, self._default)


class QwenGenerator(Generator):
    """Loads Qwen3-1.7B via transformers on first generate() call.

    Qwen3 is a unified instruction + thinking model (not a base model). It
    expects chat-template-formatted input, not raw completion-style prompts.
    We use non-thinking mode for generation (enable_thinking=False) since
    we want direct code output, not reasoning traces.

    Per the Qwen3 model card, presence_penalty=1.5 is recommended to prevent
    endless repetitions at small scale.
    """

    MODEL_ID: str = "Qwen/Qwen3-1.7B"

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device: str | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"QwenGenerator: using device={self._device}")

        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            torch_dtype="auto",
            device_map=self._device,
        )

    def generate(self, prompt: str, temperature: float) -> str:
        self._load()
        import torch

        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                repetition_penalty=1.2,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)


class GemmaGenerator(Generator):
    """Loads google/gemma-2-2b-it via transformers on first generate() call.

    Gemma 2 is an instruction-tuned model with its own chat template; we
    format inputs via apply_chat_template so no manual formatting is needed.
    float16 is used on CUDA (some Gemma variants have issues with bfloat16
    on older CUDA versions); CPU fall-back stays in float32.
    """

    MODEL_ID: str = "google/gemma-2-2b-it"

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID, dtype=dtype
        ).to(device)
        print(f"GemmaGenerator: using device={device}", flush=True)

    def generate(self, prompt: str, temperature: float) -> str:
        self._load()

        messages = [{"role": "user", "content": prompt}]
        input_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(input_text, return_tensors="pt").to(self._model.device)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.2,
        )
        generated = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        return generated


class MultiGenerator(Generator):
    """Round-robin across multiple generators.

    Each call to generate() cycles to the next generator in the list.
    ``last_generator_name`` reports which generator served the most recent
    call, for JSONL/CSV observability.
    """

    def __init__(self, generators: list[tuple[str, Generator]]):
        if not generators:
            raise ValueError("MultiGenerator requires at least one generator")
        self._generators = generators
        self._idx = 0
        self._last_name: str = ""

    def generate(self, prompt: str, temperature: float) -> str:
        name, gen = self._generators[self._idx % len(self._generators)]
        self._idx += 1
        self._last_name = name
        return gen.generate(prompt, temperature)

    @property
    def last_generator_name(self) -> str:
        return self._last_name


_FENCED_PYTHON = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
_FENCED_BARE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)


def extract_code(raw: str) -> str | None:
    """Pull the first Python function definition out of raw generator output.

    Tries in order: ```python fenced block, ``` fenced block, bare `def f(...)`.
    Returns None if no function definition can be recovered.
    """
    m = _FENCED_PYTHON.search(raw)
    if m:
        candidate = m.group(1).strip()
        if _looks_like_function(candidate):
            return candidate

    m = _FENCED_BARE.search(raw)
    if m:
        candidate = m.group(1).strip()
        if _looks_like_function(candidate):
            return candidate

    return _extract_bare(raw)


def _looks_like_function(text: str) -> bool:
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("def f(") or stripped.startswith("def f ("):
            return True
    return False


def _extract_bare(raw: str) -> str | None:
    lines = raw.splitlines()
    start = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("def f(") or stripped.startswith("def f ("):
            start = i
            break
    if start is None:
        return None

    collected = [lines[start]]
    for line in lines[start + 1:]:
        stripped = line.strip()
        if stripped == "":
            collected.append(line)
            continue
        if line[:1] in (" ", "\t"):
            collected.append(line)
            continue
        break

    while collected and collected[-1].strip() == "":
        collected.pop()

    source = "\n".join(collected)
    if not source.strip():
        return None
    return source
