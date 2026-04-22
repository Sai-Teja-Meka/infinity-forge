"""Tests for generator.py. QwenGenerator is NEVER instantiated here."""

import pytest

from infinity_forge.generator import MockGenerator, extract_code


# --- extract_code -----------------------------------------------------------

def test_extract_python_fenced():
    raw = "Here you go:\n```python\ndef f(x):\n    return x + 1\n```\nEnjoy."
    src = extract_code(raw)
    assert src is not None
    assert "def f(x):" in src
    assert "return x + 1" in src


def test_extract_generic_fenced():
    raw = "```\ndef f(x):\n    return x * 2\n```"
    src = extract_code(raw)
    assert src is not None
    assert "return x * 2" in src


def test_extract_bare_def():
    raw = "Sure, here's the code.\ndef f(x):\n    return x - 1\nthat's it"
    src = extract_code(raw)
    assert src is not None
    assert "def f(x):" in src
    assert "return x - 1" in src


def test_extract_bare_def_stops_at_non_indented_line():
    raw = (
        "def f(x):\n"
        "    return x + 1\n"
        "print('trailing garbage')\n"
    )
    src = extract_code(raw)
    assert src is not None
    assert "print" not in src
    assert "return x + 1" in src


def test_extract_returns_none_on_garbage():
    assert extract_code("just prose, no code here") is None
    assert extract_code("") is None


def test_extract_returns_none_on_whitespace_only():
    assert extract_code("   \n\n  \t\n") is None


def test_extract_prefers_python_fenced_over_bare():
    raw = (
        "Some preamble with a bare def f(x):\n"
        "    return 0\n"
        "```python\n"
        "def f(x):\n"
        "    return 99\n"
        "```\n"
    )
    src = extract_code(raw)
    assert src is not None
    assert "return 99" in src


def test_extract_preserves_multiline_body():
    raw = (
        "```python\n"
        "def f(x):\n"
        "    y = x + 1\n"
        "    z = y * 2\n"
        "    return z\n"
        "```"
    )
    src = extract_code(raw)
    assert src is not None
    assert "y = x + 1" in src
    assert "z = y * 2" in src
    assert "return z" in src


def test_extract_ignores_non_f_function():
    raw = (
        "def helper(x):\n"
        "    return x\n"
    )
    # No `def f(...)` anywhere → None
    assert extract_code(raw) is None


def test_extract_handles_trailing_blank_lines():
    raw = "```python\ndef f(x):\n    return x\n\n\n```"
    src = extract_code(raw)
    assert src is not None
    assert src.endswith("return x")


# --- MockGenerator ----------------------------------------------------------

def test_mock_generator_default_empty():
    g = MockGenerator()
    assert g.generate("anything", 0.7) == ""


def test_mock_generator_custom_default():
    g = MockGenerator(default="fallback")
    assert g.generate("unseen prompt", 0.9) == "fallback"


def test_mock_generator_canned_response():
    g = MockGenerator()
    g.set("prompt A", "response A")
    assert g.generate("prompt A", 0.7) == "response A"


def test_mock_generator_unrelated_prompt_gets_default():
    g = MockGenerator(default="def")
    g.set("prompt A", "response A")
    assert g.generate("prompt B", 0.7) == "def"


def test_mock_generator_records_calls():
    g = MockGenerator()
    g.generate("p1", 0.7)
    g.generate("p2", 1.1)
    assert g.calls == [("p1", 0.7), ("p2", 1.1)]


def test_mock_generator_key_is_stable():
    k1 = MockGenerator.prompt_key("hello")
    k2 = MockGenerator.prompt_key("hello")
    assert k1 == k2
    assert len(k1) == 16


def test_mock_generator_constructor_responses_dict():
    # Build a response map by key, not prompt
    g = MockGenerator()
    key = MockGenerator.prompt_key("x")
    g2 = MockGenerator(responses={key: "value"})
    assert g2.generate("x", 0.7) == "value"


def test_mock_generator_sequence_mode():
    g = MockGenerator(sequence=["a", "b", "c"])
    assert g.generate("p", 0.7) == "a"
    assert g.generate("p", 0.7) == "b"  # same prompt, different response
    assert g.generate("q", 0.9) == "c"


def test_mock_generator_sequence_exhausted_returns_default():
    g = MockGenerator(sequence=["a"], default="dflt")
    assert g.generate("p", 0.7) == "a"
    assert g.generate("p", 0.7) == "dflt"


# --- QwenGenerator import only ----------------------------------------------

def test_qwen_generator_importable_but_not_instantiated():
    from infinity_forge.generator import QwenGenerator
    assert QwenGenerator.MODEL_ID == "Qwen/Qwen3-1.7B"
    # Do NOT call QwenGenerator() — would load the 1.7B model.


# --- GemmaGenerator import only --------------------------------------------

def test_gemma_generator_importable_but_not_instantiated():
    from infinity_forge.generator import GemmaGenerator
    assert GemmaGenerator.MODEL_ID == "google/gemma-2-2b-it"
    assert hasattr(GemmaGenerator, "generate")
    # Do NOT call GemmaGenerator() — would load the 2B model.


# --- MultiGenerator ---------------------------------------------------------

def test_multi_generator_round_robins_across_two():
    from infinity_forge.generator import MultiGenerator

    a = MockGenerator(sequence=["a1", "a2", "a3"])
    b = MockGenerator(sequence=["b1", "b2", "b3"])
    m = MultiGenerator([("a", a), ("b", b)])

    assert m.generate("p", 0.7) == "a1"
    assert m.last_generator_name == "a"
    assert m.generate("p", 0.7) == "b1"
    assert m.last_generator_name == "b"
    assert m.generate("p", 0.7) == "a2"
    assert m.last_generator_name == "a"
    assert m.generate("p", 0.7) == "b2"
    assert m.last_generator_name == "b"


def test_multi_generator_initial_last_name_is_empty():
    from infinity_forge.generator import MultiGenerator

    a = MockGenerator(default="x")
    m = MultiGenerator([("only", a)])
    assert m.last_generator_name == ""


def test_multi_generator_single_generator_passthrough():
    from infinity_forge.generator import MultiGenerator

    inner = MockGenerator(sequence=["one", "two", "three"])
    m = MultiGenerator([("solo", inner)])

    assert m.generate("p", 0.7) == "one"
    assert m.last_generator_name == "solo"
    assert m.generate("p", 0.9) == "two"
    assert m.last_generator_name == "solo"
    assert m.generate("p", 1.1) == "three"
    assert m.last_generator_name == "solo"


def test_multi_generator_forwards_prompt_and_temperature():
    from infinity_forge.generator import MultiGenerator

    a = MockGenerator(default="a-out")
    b = MockGenerator(default="b-out")
    m = MultiGenerator([("a", a), ("b", b)])

    m.generate("prompt-1", 0.7)
    m.generate("prompt-2", 1.1)

    assert a.calls == [("prompt-1", 0.7)]
    assert b.calls == [("prompt-2", 1.1)]


def test_multi_generator_rejects_empty_list():
    from infinity_forge.generator import MultiGenerator

    with pytest.raises(ValueError):
        MultiGenerator([])
