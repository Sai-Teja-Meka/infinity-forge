"""Behavioral-richness measurement over infinity-forge survivor logs.

Criteria (fixed before the run; a survivor is "behaviorally rich" iff ALL hold):
  C1  Its fingerprint has >= 8 distinct non-error outputs across the 20 probes.
  C2  Its fingerprint differs from the fingerprint of every component that
      shares its input type (identical fingerprint = passthrough). Components
      with a different input type are probed on a different basis, so the
      comparison is vacuous and C2 passes for them.
  C3  Fewer than half its probes error (< 10 of 20 entries are __raises__:*).

Outputs: full-population counts, richness rate, and the marginal breakdown
(how many survivors fail each criterion individually) per level.
"""
import json
import sys
from collections import Counter

L1_LOG, L2_LOG, L3_LOG = sys.argv[1], sys.argv[2], sys.argv[3]

ERR = "__raises__:"


def is_err(e):
    return isinstance(e, str) and e.startswith(ERR)


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


# --- component fingerprint lookup: source -> (input_type, fingerprint) ---
lookup = {}
for rec in load(L1_LOG):
    if (rec.get("gate_result") or {}).get("accepted"):
        lookup[rec["extracted_source"]] = (rec["signature"][0], rec["fingerprint"])
l2_records = load(L2_LOG)
for rec in l2_records:
    lookup[rec["extracted_source"]] = (rec["signature"][0], rec["fingerprint"])
l3_records = load(L3_LOG)


def measure(records, label):
    n = len(records)
    fail = Counter()
    unmatched_components = 0
    rich = 0
    for rec in records:
        fp = rec["fingerprint"]
        sig_in = rec["signature"][0]

        distinct = len({json.dumps(e) for e in fp if not is_err(e)})
        c1 = distinct >= 8

        c2 = True
        for comp in rec["components"]:
            src = comp["source"]
            if src not in lookup:
                unmatched_components += 1
                continue  # different probe basis unknowable -> vacuous pass
            comp_in, comp_fp = lookup[src]
            if comp_in == sig_in and comp_fp == fp:
                c2 = False
                break

        errs = sum(1 for e in fp if is_err(e))
        c3 = errs < 10

        if not c1:
            fail["C1_low_distinct_outputs"] += 1
        if not c2:
            fail["C2_passthrough_of_component"] += 1
        if not c3:
            fail["C3_majority_probes_error"] += 1
        if c1 and c2 and c3:
            rich += 1

    print(f"== {label}: n={n}")
    print(f"   rich={rich} ({rich / n * 100:.1f}%)")
    for k in ("C1_low_distinct_outputs", "C2_passthrough_of_component", "C3_majority_probes_error"):
        print(f"   fail {k}: {fail[k]} ({fail[k] / n * 100:.1f}%)")
    print(f"   unmatched component sources: {unmatched_components}")
    return rich, n, fail


measure(l2_records, "L2 survivors")
measure(l3_records, "L3 survivors")
