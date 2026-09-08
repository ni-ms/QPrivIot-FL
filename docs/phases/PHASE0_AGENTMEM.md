# Phase 0 — Agent-Memory DP Aggregation Go/No-Go (2026-07-02)

**Verdict: CONDITIONAL GREEN → proceed to the D1+D5 bet.**

The single fact the bet rides on — *does a summable DP aggregate of memory embeddings
still retrieve like the noise-free aggregate?* — is **YES, in the regime the bet stakes
out** (many users + tiny-d embeddings + coarse buckets), and **NO** in the high-dimensional
regime. The failure boundary is governed by the release dimension (K·d) vs the number of
users N — i.e. the project's signature **d/N-crossover**, now on a NEW payload.

## Design under test — DP CENTROID MEMORY
SecAgg+Skellam only releases `Sum_i x_i`, so the shared memory must be a *summable* aggregate:
fix K data-independent buckets (random unit anchors, fixed seed); each user contributes per
bucket k `(v_ik = sum of their note-embeddings in k, c_ik = count)`; SecAgg-sum → per-bucket
`(sum-vector, count)`; DP centroid[k] = `noised_sum_v[k] / max(noised_count[k], 1)`. Shared
memory pool = K DP centroids used for retrieval.

## Faithfulness
Privacy path is **byte-identical to the FL code** — imports the project's real
`quantize / apply_distributed_skellam_noise / dequantize` from `privacy_utils.py`. Only the
payload changed (memory-note embeddings vs gradient deltas). Zero-sum SecAgg masks omitted
(cancel exactly under summation; DP-relevant part is the Skellam noise, applied verbatim).
Embedder = TF-IDF→TruncatedSVD on 20-newsgroups (11,314 docs / 20 topics), fully offline;
Phase 2 swaps in real A-MEM/Mem0 sentence embeddings. Script: `scripts/_agentmem_probe.py`.
Multi-seed (2) + reported as mean±std from the start (Phase-1 hygiene baked in).

## Results (ε = per-release single-shot Gaussian, δ=1e-5; total user-level ε ~2× under basic composition)

| Config | N | K | d | release-dim K·d | clean topic-acc | ε=8 recall@5 | ε=8 topic-acc | ε=3 recall@5 | verdict |
|--------|----|----|-----|-----|-------|--------|--------|--------|------|
| A (high-dim) | 50 | 64 | 256 | 16384 | 0.306 | 0.214 | 0.103 (≈random 0.05) | 0.144 | **RED** |
| B (tiny-d, many users) | 200 | 16 | 64 | 1024 | 0.255 | **0.813** | **0.224** (−12% of clean) | 0.675 | **GREEN** |
| C (tiny-d thesis) | 100 | 32 | 32 | 1024 | 0.291 | 0.661 | **0.267** (−8% of clean) | 0.496 | **GREEN** |

Config C ε-curve (topic-acc): ε=∞ 0.291 · ε=16 0.288 · ε=8 0.267 · ε=3 0.186 · FL-σ(2.854) 0.108
→ **graceful degradation**; DP is nearly free at ε≥8 in the tiny-d regime.

Reference ceilings: raw-doc 1-NN topic acc ≈ 0.46 (retrieval ceiling); random ≈ 0.05.

## Reading it
1. **The bet is viable.** DP centroid memory retrieves at ε=8 (and usably at ε=3) exactly where
   the bet claimed — tiny-d embeddings, many federated users. The utility side (the thing that
   could have killed the bet cheaply in 1 day) **survives**.
2. **Governed by the d/N-crossover.** B and C both have release-dim K·d = 1024 and both work;
   A at 16384 fails. Per-coordinate Skellam noise scales with the global clip norm, so cost ~
   √(K·d) — the same √d law from the QFL work, now on memory. This is the reusable analysis and
   the paper's mechanistic spine.
3. **A 3-way tradeoff to characterize (paper structure): privacy ε × memory granularity K × dim d.**
   Coarser memory (small K) is DP-robust but has lower *clean* utility (0.255 vs 0.306); finer
   memory (large K) has higher clean utility but is fragile under DP. Sweet spot is tiny-d + coarse.

## Caveats (honest)
- Utility endpoint here is retrieval/topic-fidelity of the *aggregate* vs its noise-free version —
  NOT downstream QA and NOT real memory notes. Both are Phase 2/3.
- ε accounting is single-shot Gaussian per release (×2 for the two releases). A proper Skellam
  RDP accountant for the exact discrete mechanism is a Phase-3 deliverable.
- The **positive headline endpoint — measured leakage drop (MEXTRA/MRMMIA, D5)** — is NOT yet
  shown; that is Phase 4 and remains the make-or-break for the *paper* (this phase only cleared
  the utility gate).

## Next
Green-lit to proceed: Phase 1 hygiene is already in the harness → Phase 2 (real A-MEM/Mem0 notes
+ LoCoMo/LongMemEval per-user partition) → Phase 4 leakage-drop endpoint. Recommend fixing the
operating regime near Config C (tiny-d, coarse buckets, ≥100 users) for the headline.