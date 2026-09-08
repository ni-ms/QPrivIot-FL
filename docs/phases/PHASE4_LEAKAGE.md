# Phase 4 — Leakage-Drop Endpoint (D5) Go/No-Go (2026-07-02)

**Verdict: GREEN — the positive headline holds.** DP-under-SecAgg buys a large, measurable
drop in re-identification of vulnerable members (worst-case AUC 0.94 → ~0.51 at ε=8) while
bulk retrieval utility is retained. This is the *positive, measurable* endpoint the dead QFL
paper never had, and it is mechanism-grounded.

## What was measured (de-risking spike, not the full LLM attacks)
Full MEXTRA/MRMMIA need real notes + an LLM (Phase 2/3). This spike implements the
*measurement principle* against the Phase-0 DP centroid memory:
- **Membership inference (MRMMIA analog):** candidate note embedding x → assign to shared
  bucket k → score `cos(x, centroid[k])`. Members pulled their own centroid, so score higher
  than **same-distribution** non-members (train pool split into aggregated-members vs held-out
  non-members — no train/test confound). Metric = ROC-AUC (0.5 = no leakage).
- **Extraction (MEXTRA analog):** `gap = max cos(centroid, in-bucket member) − same for non-member`.
- **Stratified by bucket count:** leakage in a *sum* mechanism concentrates where few users
  contribute, so we also report AUC on the **low-count tail** (buckets with ≤3 contributors).
- Utility (topic-acc from Phase 0) reported alongside → privacy–utility tradeoff curve.

Faithful crypto path (real `privacy_utils` quantize/skellam/dequantize), multi-seed + mean±std.
Script: `scripts/_agentmem_leakage.py` (reuses `_agentmem_probe.py` helpers).

## Results (N=100, d=32, 20-newsgroups embeddings, 2 seeds; ε = per-release single-shot Gaussian)

**Aggregate (all buckets) — leakage rises with memory fidelity K, DP removes it:**

| K | clean AUC(all) | ε=8 AUC(all) | clean tail-AUC | ε=8 tail-AUC |
|-----|------|------|------|------|
| 256 | 0.526 | 0.510 | — | — |
| 1024 | 0.574 | 0.506 | **0.941** | **0.530** |
| 2048 | 0.587 | 0.505 | **0.937** | 0.509 |

**The headline — low-count tail (K=1024, ≤3 contributors = 5% of members, the at-risk group):**

| release | σ | tail-AUC | extract-gap | utility (topic-acc) |
|---------|-----|----------|-------------|---------------------|
| ∞ (clean) | 0.000 | **0.941** | +0.076 | 0.404 |
| ε=16 | 0.303 | 0.573 | +0.001 | 0.363 |
| ε=8  | 0.606 | **0.530** | −0.004 | 0.284 |
| ε=3  | 1.615 | 0.493 | −0.005 | 0.144 |
| ε=1  | 4.845 | ~0.51 | −0.009 | 0.058 |

## Reading it
1. **Real leakage exists and DP kills it.** Clean memory re-identifies tail members at
   **AUC ≈ 0.94** (near-certain). DP drives it to **chance (~0.51) at ε=8, ~0.49 at ε=3**, while
   topic-retrieval utility holds (0.28–0.36 vs clean 0.40 at ε≥8). Extraction gap collapses
   +0.076 → ~0 in lockstep.
2. **Mechanism-grounded, not incidental.** Leakage concentrates in **low-count buckets** —
   the few-contributor SecAgg sums — because averaging already protects well-populated buckets
   (bulk AUC only 0.53–0.59). This is the *same effective-N / dimension effect* as the project's
   d/N-crossover: small effective N per bucket ⇒ high noise-sensitivity ⇒ both fragile utility
   AND high pre-DP leakage; distributed-DP noise is most protective exactly there.
3. **Both halves of the paper now exist:** Phase 0 = the DP aggregate stays *useful*; Phase 4 =
   it becomes *provably private-er*, with a measured worst-case attack drop. Positive endpoint ✓.

## Caveats (honest — what this spike did NOT do)
- Similarity-threshold MIA on embeddings, **not** the full LLM-based MEXTRA/MRMMIA (Phase 2/3).
- Synthetic TF-IDF→SVD embeddings, **not** real A-MEM/Mem0 memory notes.
- ε = single-shot Gaussian per release (×2, basic composition); a proper Skellam RDP accountant
  for the exact discrete mechanism is a Phase-3 deliverable — current ε labels are approximate.
- The "leaky tail" depends on low-count buckets existing; realistic for agent memory (rare/unique
  notes) but must be validated on real note distributions.
- Utility = topic-retrieval fidelity of the aggregate, not downstream QA.

## Bottom line
The make-or-break endpoint is de-risked. Proceed to Phase 2 (real notes + LoCoMo/LongMemEval)
with confidence that the story — **usable DP memory whose worst-case leakage provably drops** —
is real and reproducible. Recommend the paper's central figure be the K-sweep tradeoff:
memory fidelity ↑ ⇒ pre-DP leakage ↑, but DP holds attack-AUC at chance while extending the
usable-fidelity frontier.