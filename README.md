# ∞ Forge — Self-Compounding Verified Code Library

105 LLM-generated atoms compounded into 11,500 verified functions through mechanical composition — no LLM in the compounder, no hallucination in the output.

## The Idea

Small open-source models generate candidate Python functions. A six-layer cascade verifies each candidate against AST structure, static safety, type inference, sandboxed execution, source novelty, and behavioral novelty. Survivors are mechanically composed into higher-level functions: Level 2 pairs two Level 1 atoms, Level 3 pairs Level 2 with Level 1. Every composition runs back through the same cascade. There is no model in the composer — only type-matching and source rewriting. Code that proves itself compounds into more code that proves itself.

The thesis: hallucination is a complexity-budget failure, not a model failure. Keep the model inside a tight enough sandbox and it can run forever without producing garbage, because the sandbox physically will not let garbage through.

## Results (Day 11)

| Level | Count   | Source                          | Acceptance |
|-------|---------|---------------------------------|------------|
| L1    | 105     | LLM generation, cascade-verified | 21.2%      |
| L2    | 987     | Mechanical pairing of L1 × L1    | 56.0%      |
| L3    | 10,408  | Mechanical pairing of L2 × L1    | 27.2%      |
| Total | 11,500  |                                 |            |

Amplification: **109× from 105 seeds**. Every function in the library is behaviorally distinct and provably safe at its cascade level.

## Architecture

Three components, each independently testable:

**Generator** (`generator.py`, `prompts.py`, `signatures.py`)
Multi-model rotation across Gemma 2B, Phi, and Qwen via Ollama. Twenty target type signatures, one prompt per signature. The generator is deliberately weak — it only needs to occasionally produce something the cascade will accept.

**Cascade / Verifier** (`cascade.py`)
Six layers, fail-fast in order:
1. **Structural** — AST parse, single function, single argument, ≤30 lines, cyclomatic complexity ≤10.
2. **Safety** — name allowlist walk; no dunders, no imports, no `exec`/`eval`/`__import__`.
3. **Type inference** — input/output type signature recovered from probe runs.
4. **Sandbox** — subprocess isolation with kernel-enforced CPU, memory, fd, and process limits; 20 probe inputs per candidate.
5. **Source novelty** — AST canonicalization, exact-duplicate rejection.
6. **Behavioral novelty** — 20-probe output fingerprint, exact-duplicate rejection.

**Composer / Compounder** (`composer.py`)
Mechanical type-matching: pair atoms whose output type feeds the next atom's input type. Source rewriting via `compose_source` with two paths — lambda for simple compositions, nested `def` for cases that need multi-statement bodies. Every composition runs back through the full cascade. Compositions that raise on probes or return `bad_output_type` are filtered before novelty checks.

## Quick Start

```bash
git clone https://github.com/<user>/infinity-forge.git
cd infinity-forge
pip install -e .

# Generate L1 atoms (requires Ollama with gemma2:2b, phi3, qwen2.5:1.5b pulled)
forge --multi-model --iterations 500

# Compose L2 from L1
forge --compose

# Compose L3 from L2 × L1
forge --compose-l3
```

## Project Structure

```
src/infinity_forge/
  sandbox.py       # subprocess isolation, resource limits, parent-side AST gate
  cascade.py       # six-layer verifier
  generator.py     # multi-model Ollama generator loop
  composer.py      # type-matching, compose_source, L2/L3 engines
  cli.py           # forge entry point
  prompts.py       # per-signature prompts
  signatures.py    # target type signatures
  probes.py        # probe inputs for behavioral fingerprinting
  novelty.py       # AST canonicalization + behavioral dedup
  canonical.py     # AST normalization
  seeds.py         # L1 seed catalog
  inputs.py        # input generators
  forge.py         # orchestration
tests/             # 362 tests, sandbox + cascade + composer + integration
CLAUDE.md          # design decisions, non-negotiables, working style
```

## Test Suite

362 tests covering sandbox adversarial battery (21 attack vectors), cascade rejection paths, composer correctness, novelty dedup, and end-to-end integration.

```bash
pytest --timeout=120
```

## Key Technical Decisions

- **Subprocess sandbox over Docker** — kernel-enforced `setrlimit` is faster than container startup and gives stronger guarantees per-call.
- **JSONL over SQLite** — append-only, replay-friendly, no schema migrations across days.
- **AST canonicalization for source dedup** — catches semantically-identical functions that differ only in variable naming or whitespace.
- **20-probe behavioral fingerprint** — cheap to compute, strong enough to dedup functions that are textually different but compute the same thing.
- **Dual-path `compose_source`** — lambda for one-liners, nested `def` for multi-statement bodies; covers both atom shapes the generator emits.
- **Composition-readiness filter** — atoms tagged `__raises__` or `__bad_output_type__` on probes are excluded from composition pools before they can poison higher levels.

## Commit History Highlights

```
d5157a2  Day 1 — sandbox and adversarial test battery
a3f8f2d  CLAUDE.md — Day 1 context and design guardrails
991078a  Expose _ALLOWED_BUILTIN_NAMES from sandbox for cascade import
c26f030  Day 2 — static analysis cascade before sandbox execution
13d7aa9  Day 3 — the generator loop (mocked in tests, real on Colab)
0e6f7a0  Fix QwenGenerator to use chat template and repetition penalty
9b705ad  Restore MockGenerator.set/prompt_key and f-name requirement
3a9c7fd  Day 4 — Layer 6 behavioral fingerprinting, seeds, prompt refinement
becc1f3  Day 5 — observability for the overnight measurement run
ea3cad1  Day 6 — AST canonicalization, seed and prompt fixes
7c98435  Day 7 — second generator model (Gemma 2 2B), remove dead signature
3ab99d7  Day 8 — third generator model (Phi-3.5-mini-instruct)
d29ae8d  Swap Phi-3.5-mini for SmolLM2-1.7B-Instruct
978b511  Day 9 — bridge signatures + Level 2 composition engine
4e1084c  Day 10 — L1-L2 compatibility hardening
8baca56  Revert Day 10 prompt hardening (keep composer infrastructure)
9e5bf41  Fix nested-def return bug in compose_source
7062231  Level 3 composition engine + composer refactor
```

## License

MIT
