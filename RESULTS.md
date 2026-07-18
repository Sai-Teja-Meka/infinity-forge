# RESULTS — Behavioral-Richness Measurement

**Date of measurement:** 2026-07-17
**Measured by:** `scripts/measure_richness.py` (committed in this repo), single run, criteria fixed before execution.
**Environment:** Python 3.12.7, Windows 11, stdlib only. Input artifacts: the Day-11 run logs (Colab, 2026-04-25) committed under `artifacts/day11/`.

## What was measured

Every Level-2 and Level-3 composition survivor from the final (Day-11) run — the full population, no sampling. Each survivor's stored 20-probe behavioral fingerprint (recorded at cascade time) was analyzed statically; no functions were re-executed.

| Population | n |
|---|---|
| L2 survivors | 987 |
| L3 survivors | 10,408 |

## Criteria (as implemented)

A survivor is **behaviorally rich** iff all three hold:

- **C1 — Output diversity:** its fingerprint contains ≥ 8 distinct non-error outputs across the 20 probes (not constant, not collapsing to a few values).
- **C2 — Not a passthrough:** its fingerprint differs from the fingerprint of every component that shares its input type. (Components with a different input type are probed on a different basis; the comparison is vacuous and C2 passes.)
- **C3 — Not error-dominated:** fewer than half of its probes error (< 10 of 20 entries are `__raises__:*`).

These criteria are heuristic and were fixed before the run; no thresholds were adjusted after seeing results. An earlier informal evaluation of richness existed but its method and artifacts were not preserved; this is a new operationalization and supersedes any previously quoted figures.

## Results

| Level | n | Rich | Rate |
|---|---|---|---|
| **L2** | 987 | 589 | **59.7%** |
| **L3** | 10,408 | 4,867 | **46.8%** |

### Marginal breakdown (failures per criterion, independently counted)

| Criterion | L2 | L3 |
|---|---|---|
| C1 — low output diversity | 395 (40.0%) | 5,515 (53.0%) |
| C2 — passthrough of a component | 0 (0.0%) | 0 (0.0%) |
| C3 — majority of probes error | 3 (0.3%) | 52 (0.5%) |

Unmatched component sources: 0 at both levels (every component resolved to a stored fingerprint).

## Findings

1. **Richness failure is almost entirely an output-diversity failure (C1).** Survivors that aren't rich are overwhelmingly functions whose outputs collapse to a small set of values across probes — not passthroughs, not error factories.
2. **C2 = 0 is structural, not luck.** The cascade's Layer 6 (behavioral-novelty dedup) already rejects candidates whose fingerprint duplicates an accepted function of the same signature, so passthrough compositions cannot survive to be measured. The measurement confirms the layer does its job. (Fingerprints can legitimately repeat *across* signatures — 5 such pairs exist among the 114 L1 atoms — because probe bases differ per input type.)
3. **Correctness ≠ richness.** All 11,395 composition survivors are verified for correctness at their cascade level; about half additionally exhibit behaviorally rich output under these criteria. The two properties are different claims and the README states them separately.

## Reproduce

```bash
python scripts/measure_richness.py \
  <(gunzip -c artifacts/day11/l1.jsonl.gz) \
  <(gunzip -c artifacts/day11/l2.jsonl.gz) \
  <(gunzip -c artifacts/day11/l3.jsonl.gz)
```
