# Phase 3 — Skellam RDP Accountant (2026-07-02)

Replaces the approximate single-shot Gaussian ε labels with the **exact discrete-mechanism
guarantee**. This is the piece reviewers scrutinise most; it is now rigorous.

## Implementation
`skellam_rdp_epsilon(sigma, clip_norm, scale, dim, delta, releases, l1_sensitivity)` in
`qpriviot_fl/privacy_utils.py`, from **Agarwal, Kairouz & Liu (NeurIPS 2021), Theorem 3.5**:

    ε_RDP(α) ≤ α·Δ2²/(2μ) + min{ ((2α−1)·Δ2² + 6·Δ1)/(4μ²),  3·Δ1/(2μ) }

with μ = per-coordinate released variance = (σ·C·s)², Δ2 = C·s, Δ1 ≤ √dim·Δ2, s = range_max/bound.
The parametrisation was verified against the paper's Def 3.1 (PMF e^{-μ}I_k(μ), E‖noise‖²=dμ):
**μ is the per-coordinate variance** (each Poisson has mean μ/2), which matches this project's
`skellam_noise(variance=…)` exactly. Composition over `releases` is additive in RDP; (ε,δ)
conversion via ε = min_α [RDP(α) + ln(1/δ)/(α−1)]. Report: `scripts/_skellam_accounting.py`.

## Result 1 — corrected ε (the labels were optimistic AND out-of-range)
The phase tables labelled σ via σ = √(2 ln(1.25/δ))/ε — the **classic Gaussian bound, valid only
for ε ≤ 1**, so the labels were invalid at the ε=8/16 operating points. Rigorous Skellam-RDP ε
(δ=1e-5, operating point K=32,d=32):

| printed label | σ | classic (invalid >1) | Gauss-RDP | **Skellam 1-release** | **Skellam 2-release** |
|-----|------|------|------|------|------|
| "ε=16" | 0.303 | 15.99 | 22.09 | 22.09 | 33.30 |
| "ε=8"  | 0.606 | 7.99  | 9.28  | **9.28** | **13.93** |
| "ε=3"  | 1.615 | 3.00  | 3.16  | 3.16 | 4.60 |
| FL-σ   | 2.854 | 1.70  | 1.74  | 1.74 | 2.50 |

So the "ε=8" experiments are rigorously **ε≈9.3** (one release) or **ε≈13.9** (sum-vector + counts
composed). Conclusions from the utility/leakage phases are **monotone in σ**, so every ordering,
retention %, and AUC-drop is unchanged — only the ε axis re-labels (upward: privacy was slightly
weaker than the optimistic labels implied, now rigorously bounded).

## Result 2 — discretisation is provably free at this resolution
Skellam ε vs Gaussian-RDP ε as the quantiser resolution grows (σ=0.606, 1 release):

| range_max | scale | Skellam ε | Gauss-RDP ε | surcharge |
|-----------|-------|-----------|-------------|-----------|
| 1e2 | 10    | 9.7694 | 9.2837 | 0.4857 |
| 1e3 | 100   | 9.2854 | 9.2837 | 0.0017 |
| 1e4 | 1000  | 9.2837 | 9.2837 | 0.0000 |
| 1e6 | 1e5   | 9.2837 | 9.2837 | 0.0000 |

At the project's **range_max = 1e6 the discrete Skellam mechanism equals the Gaussian to 4+
decimals** — the discretisation is free (matches the paper's 1+O(1/μ) claim), now *proven* for
our configs rather than assumed. The surcharge only bites at coarse resolution (≤ a few hundred).

## Recommendation for the paper
- **Re-label all phase tables** with the rigorous 2-release Skellam ε (or switch to the tighter
  single-release design below and use the 1-release column).
- **Free privacy win:** concatenate the sum-vector and count vector into ONE clipped Skellam
  release instead of two → single RDP composition → ε≈9.3 instead of 13.9 at σ=0.606. Minor
  refactor of the two `secagg_skellam_release` calls into one.
- Report Result 2 as a small figure — it justifies the integer/SecAgg quantisation as privacy-free.

## Joint vs separate release — empirical check (corrects the "free win" framing)
Implemented the single-release design (`--joint` in `_longmemeval_probe.py`,
`build_centroids_joint`): concatenate [sum-vector, counts] per user, clip to ONE L2 norm, one
Skellam release → one RDP composition. Evidence-recall@5, K=32/d=32, 2 seeds, at matched σ:

| σ | rigorous ε (sep 2-rel) | recall (sep) | rigorous ε (joint 1-rel) | recall (joint) |
|-------|------|------|------|------|
| 0.303 | 33.3 | 0.555 | 22.1 | 0.549 |
| 0.606 | 13.9 | 0.547 | 9.3  | 0.514 |
| 1.615 | 4.6  | 0.466 | 3.2  | 0.373 |
| 2.854 | 2.5  | 0.391 | 1.7  | 0.300 |

**Honest reading (not a dramatic free win):** at matched σ, joint has a *better* ε (fewer
compositions) but slightly *lower* utility — the joint release shares one clip norm and one
quantiser bound across the count and vector channels, and the integer counts dominate the bound,
costing the sum-vector some resolution. On the privacy–utility *frontier* (recall vs rigorous ε)
the two are essentially on top of each other: interpolating the separate curve to ε=9.3 gives
≈0.507 vs joint's 0.514 — joint is marginally better and comes with a cleaner single-composition
guarantee. **Verdict on joint:** on par, not a dramatic win; the shared clip/quantiser bound lets the
integer counts starve the vector channel.

## The actual free win — VECTOR-ONLY release
The retrieval centroid is `normalize(sum_v[k] / count[k])`, and dividing by the scalar count
then L2-normalising **cancels the count**: `normalize(sv/sc) = normalize(sv)`. So the count
channel never affects retrieval direction — it did nothing for utility in *any* run, yet the
separate design paid for a second Skellam release (ε 9.3 → 13.9) to protect it. Releasing ONLY
the summed vector (`--release veconly`, 1 composition) gives, at matched σ:

| σ | separate (2-rel) ε / recall | **vec-only (1-rel) ε / recall** |
|-------|------|------|
| 0.606 | 13.9 / 0.547 | **9.3 / 0.547** |
| 1.615 | 4.6 / 0.466 | **3.2 / 0.466** |
| 2.854 | 2.5 / 0.391 | **1.7 / 0.391** |

Recall is **identical to the last decimal** (same seeded sum-vector noise; the count only ever
cancelled), so vector-only **strictly dominates**: same utility, ~1.5× tighter ε, one composition.
This — not the joint concatenation — is the recommended design for cosine retrieval.
Caveat: if bucket *occupancy* must also be privatised (or actual mean magnitudes are needed), add
a cheap separate low-sensitivity count/histogram release; for pure retrieval it is unnecessary.

## Remaining
Full `s`-variant stress test (247k notes); LLM-distilled A-MEM/Mem0 notes; full LLM-driven
MEXTRA/MRMMIA attacks. Accounting is paper-ready; recommended mechanism = vector-only release (ε≈9.3).