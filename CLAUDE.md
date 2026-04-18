# CLAUDE.md — ∞ Forge

This file is loaded by Claude Code at the start of every session in this repo. Read it fully before proposing any changes.

## What this project is

∞ Forge is a **self-compounding verified Python function library**. A generator produces candidate Python functions; a multi-layer cascade accepts or rejects each candidate; accepted atoms become seeds for the next round of generation. The library grows indefinitely inside a constraint envelope that makes hallucination structurally impossible to accept — every node in the graph is provably correct at its cascade level.

The project is built around a central thesis: **hallucination is a function of complexity budget exceeded, not a function of the model itself.** A small model kept strictly inside a tight sandbox can generate forever without producing garbage, because the sandbox physically won't let garbage through. Instead of making the model bigger to reduce error rates, we make the sandbox tighter and let the model run in a loop.

∞ Forge is the first domain (code) of what will eventually be a multi-domain forge — numeric first, then strings, collections, and beyond. Each domain compounds on the primitives of earlier domains. The output is simultaneously (a) a usable library people can import and (b) cognitive-DNA substrate for persona-based systems being built in a sibling project called **Combo**.

## Current state

**Day 1 is complete and committed as `d5157a2`.** The sandbox exists, is verified against 21 adversarial attack vectors plus 17 correctness tests, and is ready to serve as the lowest layer of the cascade.

Day 2 onward is not yet built. Do not propose Day 2 or later work unless the current session explicitly says we are working on Day 2.

### What's in the repo

- `src/infinity_forge/sandbox.py` — the only production code. Implements `run_in_sandbox(source, input, timeout_ms=500, memory_mb=100)`. Subprocess-based isolation with kernel-enforced resource limits (CPU, address space, file descriptors, process count, file size), restricted builtins, JSON-over-stdio communication with the child, parent-side AST validation before subprocess spawn.
- `tests/test_sandbox_basic.py` — 17 correctness tests (1 unit + 12 parametrized + 4 specific).
- `tests/test_sandbox_adversarial.py` — 21 adversarial tests: infinite loops, recursion, memory exhaustion, import escapes, fork bombs, file read/write, network, ctypes crashes, oversized output, non-JSON output, source validation, input validation, zombie cleanup, 100-run stress.
- `pyproject.toml` — setuptools build, Python ≥3.12, pytest + pytest-timeout as dev deps, project-wide pytest `timeout=3` default.
- Standard scaffolding: `.gitignore`, `.python-version` (3.12.3), `README.md`, `tests/__init__.py`.

### Environment

- WSL2 Ubuntu on Windows (kernel 6.6.87.2-microsoft-standard-WSL2).
- Python 3.12.3 via system `python3`.
- `.venv/` at the project root, created with `python3 -m venv` after `python3.12-venv` was installed via `sudo apt install`.
- pytest 9.0.3, pytest-timeout 2.4.0.
- Package installed in editable mode via `pip install -e .`.
- All work is done inside WSL2's native filesystem at `~/projects/infinity-forge`, not through the `/mnt/c/` Windows mount.

### Test performance baseline (Day 1)

- Happy-path `run_in_sandbox` call: ~22ms end-to-end on this hardware.
- Python subprocess cold start on WSL2: 30-50ms (dominates cascade cost).
- Full test suite (38 tests): 5.66-7.32s depending on caches.
- Per-test average: ~150-200ms including subprocess teardown overhead.

These numbers inform parallelization design for later days. Subprocess startup is the bottleneck, not function execution — Day 4's generator-loop architecture should account for this.

## Design decisions that are non-negotiable

These decisions were made deliberately during Day 1. Do not revisit or "simplify" them without explicit discussion.

### The default timeout is 500ms, not 50ms

The original Day-1 spec called for a 50ms default. That was wrong for WSL2 because Python subprocess startup alone is 30-50ms on this platform, so a 50ms default guarantees false-positive timeouts on correct code. 500ms leaves ~440ms of real execution budget after startup and child-script overhead — generous for pure functions, tight enough to catch pathological ones quickly. The docstring in `sandbox.py` explains this in full. Do not lower this default without benchmarking first.

### The child catches `BaseException`, not `Exception`

`SystemExit` and `KeyboardInterrupt` inherit from `BaseException`, not `Exception`. A malicious payload can `raise SystemExit(0)` to escape an `except Exception` clause, exit the child with a zero return code, and leave the parent reporting `status: "crash"` instead of `status: "error"`. Catching `BaseException` closes this hole. Do not narrow this to `Exception`.

### The kill-then-terminate fallback is dead code, and that's intentional

The timeout path sends SIGKILL first (uncatchable), then SIGTERM (catchable) as fallback if SIGKILL didn't take. On a healthy Linux kernel SIGKILL always succeeds, so the SIGTERM branch is dead code. It's retained as defense-in-depth for exotic failure modes (stuck in D-state on a broken filesystem, ptrace-attached by a debugger) where SIGKILL can be delayed. SIGTERM won't actually help in those cases either, but trying every stdlib-exposed signal before giving up is the right posture. The comment in `sandbox.py` documents this. Do not remove the fallback.

### The child namespace has exactly 23 allowed builtins

The allowlist is: `abs, min, max, sum, len, range, sorted, reversed, enumerate, zip, map, filter, all, any, round, int, float, str, list, tuple, dict, set, bool` plus constants `True, False, None`. This is intentionally tight. Do not add to this list casually — every addition is a potential sandbox escape vector. If Day 2+ work requires additional builtins, that decision should be a conscious design discussion, not a silent expansion.

### No `__import__`, no `exec`, no `eval` in the child namespace

These three builtins are specifically excluded so that imported-module attacks fail at the name-lookup layer, not somewhere downstream. Even if later layers add stricter checks, the child namespace should never regain these.

### Environment is scrubbed before subprocess spawn

The child runs with `env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"}` — no `HOME`, no `USER`, no parent env leaks. This prevents secrets in parent environment from reaching sandboxed code and removes ambient-authority attack surface. If future work needs to set additional env vars, add them explicitly to this dict; don't switch to `env=None` (inherits parent env).

### Parent-side source validation happens before subprocess spawn

`_validate_source` uses `ast.parse` to confirm the source has exactly one top-level `FunctionDef` and no other top-level statements. This rejection happens without spawning any subprocess, returning `runtime_ms=0`. This is not a security check — it's a cheap fast-path. Day 2's Layer 1 AST walker will add real security-level static analysis on top of this, but the validator's sub-100ms rejection path for obviously-bad input stays.

## What NOT to touch without thinking

- **`sandbox.py` is frozen.** It is the foundation of everything. Changes to sandbox.py should happen only if: (a) we identify a genuine security flaw, (b) performance profiling shows a specific bottleneck, or (c) a later cascade layer requires new behavior from the sandbox. "Cleaning up" or "refactoring" sandbox.py for style reasons is not acceptable — every line has been adversarially tested.
- **The `_CHILD_SCRIPT` heredoc** in `sandbox.py` is particularly sensitive. It runs inside the subprocess and handles every untrusted execution. Any edit to it needs re-verification against the full 21-test adversarial battery.
- **`resource.setrlimit` calls in `_make_preexec`** must run in the child process (via `preexec_fn`), never in the parent. The parent's limits should remain unconstrained.

## What's coming in later days

For context, not for implementation. Do not start these unless the session explicitly says so.

- **Day 2:** AST-level static safety cascade (Layer 1: parse, single-function, banned nodes; Layer 2: name/attribute allowlist walk). Rejects ~40-45% of generator output before the sandbox ever spawns.
- **Day 3:** Generator loop. Wire Qwen3-1.7B via vLLM/lmdeploy, cascade-filter candidates, log acceptances. No novelty layer yet.
- **Day 4:** Novelty layers (source embedding + behavioral fingerprint), SQLite storage, parallelization.
- **Day 5:** Web visualization — live graph of accepted atoms streamed over WebSocket.
- **Day 6-7:** Overnight run to 10,000 verified numeric atoms.

## Working style for this repo

- **Every file creation and non-trivial bash command gets approved one at a time.** Don't batch-approve with "yes, allow all edits during this session." Day 1 established this discipline; Day 2 continues it.
- **Commit discipline:** one commit per meaningful unit of work. Commit messages in imperative mood. Co-Authored-By trailer should name the current Claude model (check `/model` at session start — don't hardcode an older one).
- **No sudo inside Claude Code.** If a command requires sudo (like the one-time `apt install python3.12-venv` during Day 1 bootstrap), the user runs it manually outside the Claude Code session and reports back.
- **Tests are ground truth.** If a test fails, don't patch the test to match the code. Understand why the behavior diverged from the expectation, then decide whether the code is wrong or the test's expectation was wrong. The adversarial tests in particular should never be weakened to make them pass.
- **When in doubt, ask.** This project is security-critical by design. Asking for clarification is cheaper than rolling back a flawed commit.

## Author context

The human working on this is **Sai Teja Meka**, AI/ML Engineer. He's building ∞ Forge as the metabolism layer of a larger system called **Combo**, which is a multi-persona LLM platform inspired by the power-system architecture of the novel *Infinite Mana in the Apocalypse*. The forge produces verified atoms; Combo's personas metabolize those atoms into cognitive DNA. Understanding this framing isn't required to write code, but it explains why design decisions lean toward **compounding systems rather than one-shot solutions**, and why correctness-by-construction matters more than raw throughput.

If a decision could go either way on performance-vs-correctness, default to correctness. The forge will run overnight unattended; every corner cut becomes a 3am problem later.
