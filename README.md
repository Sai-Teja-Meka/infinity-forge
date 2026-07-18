# ∞ Forge — Self-Compounding Verified Code Library

9 hand-written seeds → 114 LLM-generated atoms → **11,509 functions verified for correctness** through a six-layer cascade, with no LLM in the compounder. Of the L2/L3 composition survivors, **59.7% / 46.8% additionally exhibit behaviorally rich output** — measured over the full population, method and marginals in [RESULTS.md](RESULTS.md).

## The Idea

Small open-source models generate candidate Python functions. A six-layer cascade verifies each candidate against AST structure, static safety, type inference, sandboxed execution, source novelty, and behavioral novelty. Survivors are mechanically composed into higher-level functions: Level 2 pairs two Level 1 atoms, Level 3 pairs Level 2 with Level 1. Every composition runs back through the same cascade. There is no model in the composer — only type-matching and source rewriting. Code that proves itself compounds into more code that proves itself.

The thesis: hallucination is a complexity-budget failure, not a model failure. Keep the model inside a tight enough sandbox and it can run forever without producing garbage, because the sandbox physically will not let garbage through.

## Results (Day 11 run — artifacts committed)

| Level | Count | Source | Acceptance |
|-------|--------|---------------------------------|------------|
| Seeds | 9 | hand-written catalog (`seeds.py`) | — |
| L1 | 114 | LLM generation, cascade-verified (114/509 candidates) | 22.4% |
| L2 | 987 | mechanical pairing of L1 × L1 (987/1,761) | 56.0% |
| L3 | 10,408 | mechanical pairing of L2 × L1 (10,408/38,302) | 27.2% |
| **Total** | **11,509** | | |

Amplification: **~101× from 114 atoms** (11,395 composed functions). Every function is verified at its cascade level, and behaviorally distinct from every accepted function of the same signature. Behavioral *richness* is a separate, stronger property: 59.7% of L2 and 46.8% of L3 survivors meet it ([RESULTS.md](RESULTS.md)).

The full run logs (survivors + fingerprints, gzipped) are committed under [`artifacts/day11/`](artifacts/day11/), with a readable 200-function sample in [`artifacts/sample_200.jsonl`](artifacts/sample_200.jsonl). The complete raw log set (including rejection logs, ~84 MB) is published as a GitHub Release asset.

## Architecture

Three components, each independently testable:

**Generator** (`generator.py`, `prompts.py`, `signatures.py`)
Multi-model rotation across Qwen3-1.7B, Gemma-2-2B-it, and SmolLM2-1.7B via `transformers` (runs executed on Colab GPU). Twenty target type signatures, one prompt per signature. The generator is deliberately weak — it only needs to occasionally produce something the cascade will accept.

**Cascade / Verifier** (`cascade.py`)
Six layers, fail-fast in order:
1. **Structural** — AST parse, single function, single argument, ≤30 lines, cyclomatic complexity ≤10.
2. **Safety** — name allowlist walk; no dunders, no imports, no `exec`/`eval`/`__import__`.
3. **Type inference** — input/output type signature recovered from probe runs.
4. **Sandbox** — subprocess isolation with kernel-enforced CPU, memory, fd, and process limits; 20 probe inputs per candidate.
5. **Source novelty** — AST canonicalization, exact-duplicate rejection.
6. **Behavioral novelty** — 20-probe output fingerprint, per-signature exact-duplicate rejection.

**Composer / Compounder** (`composer.py`)
Mechanical type-matching: pair atoms whose output type feeds the next atom's input type. Source rewriting via `compose_source` with two paths — lambda for simple compositions, nested `def` for cases that need multi-statement bodies. Every composition runs back through the full cascade. Compositions that raise on probes or return `bad_output_type` are filtered before novelty checks.

## Quick Start

```bash
git clone https://github.com/Sai-Teja-Meka/infinity-forge.git
cd infinity-forge
pip install -e .

# Generate L1 atoms (downloads Qwen3-1.7B / Gemma-2-2B via transformers; GPU recommended)
forge --multi-model --iterations 500

# Compose L2 from L1
forge --compose

# Compose L3 from L2 × L1
forge --compose-l3

# Reproduce the richness measurement from the committed artifacts
python scripts/measure_richness.py \
  <(gunzip -c artifacts/day11/l1.jsonl.gz) \
  <(gunzip -c artifacts/day11/l2.jsonl.gz) \
  <(gunzip -c artifacts/day11/l3.jsonl.gz)
```

## Project Structure

```
src/infinity_forge/
  sandbox.py       # subprocess isolation, resource limits, parent-side AST gate
  cascade.py       # six-layer verifier
  generator.py     # multi-model transformers generator loop
  composer.py      # type-matching, compose_source, L2/L3 engines
  cli.py           # forge entry point
  prompts.py       # per-signature prompts
  signatures.py    # target type signatures
  probes.py        # probe inputs for behavioral fingerprinting
  novelty.py       # AST canonicalization + behavioral dedup
  canonical.py     # AST normalization
  seeds.py         # L1 seed catalog (9 seeds)
  inputs.py        # input generators
  forge.py         # orchestration
scripts/
  measure_richness.py  # behavioral-richness measurement (see RESULTS.md)
artifacts/
  day11/           # final-run survivor logs + fingerprints (gzipped)
  sample_200.jsonl # readable sample: 20 L1 + 60 L2 + 120 L3
tests/             # 362 tests, sandbox + cascade + composer + integration
RESULTS.md         # richness measurement: method, criteria, marginals
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

## Limitations

- **Correctness ≠ utility.** "Verified" means passed the six-layer cascade at its level: structurally safe, type-consistent, sandbox-clean, and novel. It does not mean the function is useful for any particular task.
- **Richness criteria are heuristic.** The behavioral-richness measurement (RESULTS.md) operationalizes "rich" as output diversity + non-passthrough + non-error-dominated over 20 fixed probes. Different criteria would give different rates; the criteria and marginals are published so the number can be argued with.
- **Single domain, single argument.** The current forge produces single-argument numeric/collection functions from 20 signatures. Multi-domain compounding is the roadmap, not the artifact.
- **Composition depth is 3.** L4+ is untested; the acceptance-rate falloff (56% → 27%) suggests richness pressure grows with depth.

## Commit History Highlights

```
d5157a2  Day 1 — sandbox and adversarial test battery
c26f030  Day 2 — static analysis cascade before sandbox execution
13d7aa9  Day 3 — the generator loop (mocked in tests, real on Colab)
3a9c7fd  Day 4 — Layer 6 behavioral fingerprinting, seeds, prompt refinement
978b511  Day 9 — bridge signatures + Level 2 composition engine
7062231  Level 3 composition engine + composer refactor
```

## License

MIT — see [LICENSE](LICENSE).
