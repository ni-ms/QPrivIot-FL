# Phase 2 — Real Sentence Embeddings + Projection-Dim Sweep (2026-07-02)

**Verdict: GREEN — the Phase-0 utility crossover and Phase-4 leakage-drop both SURVIVE real
embedding geometry, and the "tiny-d" design lever is validated and quantified.**

## What changed vs Phase 0/4
Swapped the synthetic TF-IDF→SVD embedder for a **real sentence encoder**
(`all-MiniLM-L6-v2`, 384-d) on the same 20-newsgroups corpus, then made the **projection
dimension d an explicit design knob** (PCA to d ∈ {32,64,128,384}). This isolates the biggest
geometry change (real embeddings) and tests the tiny-d thesis directly, since real embeddings
are NOT natively tiny-d. Raw MiniLM vectors are encoded once and cached
(`experiment_results/_st_20ng_raw.npz`); `--embedder st` in both probe scripts. Multi-seed(2)+mean±std.

Everything else identical and faithful: real `privacy_utils` crypto path, DP centroid memory,
same-distribution MIA, low-count-tail stratification.

## Utility (N=100, K=32; real MiniLM; ε = single-shot Gaussian per release)

| proj d | raw-doc ceiling | clean centroid | ε=8 topic-acc | **DP retention @ε8** | ε=8 recall@5 |
|--------|-----------------|----------------|---------------|----------------------|--------------|
| 32  | 0.601 | 0.336 | 0.313 | **93%** | 0.729 |
| 64  | 0.635 | 0.371 | 0.303 | 82% | 0.635 |
| 128 | 0.660 | 0.379 | 0.281 | 74% | 0.571 |
| 384 | 0.661 | 0.341 | 0.194 | 57% | 0.436 |

Higher d → better *clean* utility (ceiling 0.60→0.66) but worse *DP* retention (93%→57%) — the
√d cost, now on real embeddings. Real embeddings also lift the ceiling well above TF-IDF (0.66 vs 0.46).

## Leakage (N=100, K=1024 leaky-tail config; real MiniLM)

| proj d | clean AUC(all) | clean tail-AUC | ε=8 tail-AUC | ε=3 tail-AUC | extract-gap clean→ε8 |
|--------|----------------|----------------|--------------|--------------|----------------------|
| 32  | 0.652 | 0.948 | 0.520 | 0.519 | 0.120 → 0.015 |
| 64  | 0.699 | 0.966 | 0.519 | 0.483 | 0.174 → 0.011 |
| 128 | 0.762 | 0.985 | 0.517 | 0.522 | 0.213 → 0.013 |
| 384 | 0.787 | 0.985 | 0.526 | 0.504 | 0.241 → 0.006 |

Real embeddings are **more leaky** than TF-IDF (clean tail-AUC 0.95–0.99 vs 0.94; clean all-AUC
up to 0.79 vs 0.59) because they carry more identifying information. **DP still collapses the
tail to chance (~0.52) at ε=8 for every d**, and the extraction gap collapses ~15–40× in lockstep.

## The Phase-2 headline (both sweeps together — the paper's central figure)
As projection dim d rises: **pre-DP leakage exposure rises** (tail-AUC 0.95→0.99, all-AUC
0.65→0.79, extract-gap 0.12→0.24) **AND DP utility retention falls** (93%→57%). Both axes point
the same way ⇒ **project real embeddings to tiny-d (≈32) before DP aggregation**: near-maximal
utility retention under DP *and* minimal identifying information for an attacker to exploit,
while DP guarantees attack-AUC ≈ chance. The tiny-d thesis is not just compatible with real
embeddings — it is *favored* on both privacy and utility.

## Caveats (what Phase 2 still did NOT do — this remains a spike, not the paper)
- Corpus is still 20-newsgroups (real embeddings, real documents) — **not yet a conversational
  agent-memory benchmark** (LoCoMo / LongMemEval). Per-user = topic-Dirichlet, not real users.
- Notes = raw documents, **not LLM-distilled A-MEM/Mem0 memory notes** (needs an LLM).
- Attack = same-distribution threshold MIA + centroid reconstruction, **not the full
  LLM-driven MEXTRA/MRMMIA** pipelines.
- ε = single-shot Gaussian per release (×2, basic composition); **Skellam RDP accountant** for
  the exact discrete mechanism still owed (Phase 3).
- Utility = topic-retrieval fidelity, not downstream QA.

## Bottom line
The bet now survives its three biggest reality checks — real crypto path (P0), a measurable
positive privacy endpoint (P4), and real embedding geometry with a validated tiny-d lever (P2).
Remaining work is the actual paper build: real conversational-memory data + LLM-distilled notes
+ real MEXTRA/MRMMIA + Skellam RDP accountant. Recommended operating point for the headline:
tiny-d (d≈32) projected memory, coarse buckets for the utility story, fine buckets to expose the
leakage-drop, tradeoff curve in between.