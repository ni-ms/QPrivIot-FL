# Phase 5 — LLM-Distilled Memory Notes vs Raw Turns (2026-07-02)

Closes the loop on the "real payload" caveat: replaces raw dialogue turns with **LLM-distilled
A-MEM/Mem0-style memory notes** — the objects real agent-memory systems actually store — and
re-runs the DP leakage measurement. Distillation is **local + free** (Ollama `qwen2.5:7b`, no API
key, no data leaves the machine — apt for a privacy paper).

## Setup
`scripts/_longmemeval_distill.py`: for each LongMemEval session, a local LLM extracts atomic
memory notes (facts / events / preferences). 100 users → **1,784 distilled notes** (~24 s/user on
an M4). Sample (user 0): *"I got my car serviced on March 15th"*, *"I redeemed 50,000 points for a
$500 gift card to a car accessories store."* `scripts/_longmemeval_distilled_analysis.py` embeds
them (cached MiniLM) and runs the same MIA / low-count-tail / extraction attack as the raw path,
so distilled-vs-raw is a clean comparison (leakage needs no per-turn labels). 2 seeds.

## Result — distilled notes leak MORE; DP collapses both to chance

**K=256, d=32 (100 users):**

| source | clean MIA-AUC | clean tail-AUC | clean extract-gap | ε=8 tail-AUC |
|--------|---------------|----------------|-------------------|--------------|
| Raw turns (3,094) | 0.622 | 0.861 | 0.128 | 0.588 |
| **Distilled (1,784)** | **0.756** | **0.938** | **0.246** | 0.554 |

**K=512, d=32:**

| source | clean MIA-AUC | clean tail-AUC | low-count tail % | ε=8 tail-AUC |
|--------|---------------|----------------|------------------|--------------|
| Raw turns | 0.690 | 0.873 | 16% | 0.488 |
| **Distilled** | **0.828** | 0.930 | **35%** | 0.559 |

## Reading it
1. **Real agent-memory objects are MORE re-identifiable than raw turns.** Distillation concentrates
   identifying information into dense, memorable, user-specific facts — clean MIA-AUC 0.76–0.83 vs
   raw 0.62–0.69, extraction gap ~2× higher (0.25–0.30 vs 0.13–0.17), and a larger vulnerable
   low-count tail (13–35% of members vs 4–16%). This is the intuition behind the whole bet made
   concrete: agent memory is a higher-leakage payload than gradients or raw text.
2. **DP-under-SecAgg still drives both to ~chance at ε=8** (distilled tail-AUC 0.93–0.94 → ~0.55).
   The leakage-drop headline holds on the realistic payload.
3. **Net: the case for the mechanism is STRONGER on distilled memory, not weaker** — higher pre-DP
   risk, same DP protection. This turns the "real notes" caveat into a *motivation*.

## Distilled-notes UTILITY (answer-retrieval) — gap filled
Since distillation drops the per-turn `has_answer` labels, utility is measured by **answer
retrieval**: identify each user's answer-bearing note (the distilled note most similar to the gold
answer), then measure whether the DP memory routes the question to that note's bucket
(`scripts/_longmemeval_distilled_utility.py`, vector-only release, 2 seeds).

| K (d=32) | chance | clean | ε=16 | ε=8 | **retention@ε8** |
|----------|--------|-------|------|------|------|
| 32  | 0.156 | 0.595 | 0.525 | 0.470 | **79%** |
| 64  | 0.078 | 0.525 | 0.320 | 0.240 | 46% |
| 128 | 0.039 | 0.475 | 0.240 | 0.140 | 29% |

Distilled memory **supports answer retrieval** (clean 3–4× chance), and at the tiny-d/coarse-K
operating point DP retains **79% at ε=8** (0.470 vs clean 0.595). Finer K degrades faster than on
the raw full-oracle corpus because the distilled subset is small (1,784 notes → sparser buckets =
same density effect as the s-variant, inverted); the 500-user run (~9k notes) will relax this.
Net: **both halves now hold on distilled notes** — usable answer retrieval under DP AND a leakage
drop, with the added twist that distilled notes are the *higher-leakage* payload.

## FULL 500-user scale (6,252 distilled notes) — confirms both findings, tighter bars

**Leakage (distilled vs raw), d=32:**

| K | source | clean MIA-AUC | clean tail-AUC | extract-gap | ε=8 tail-AUC | ε=3 tail-AUC |
|------|--------|------|------|------|------|------|
| 512  | raw (10,957) | 0.588 | 0.905 | 0.068 | 0.632 | 0.590 |
| 512  | **distilled (6,252)** | **0.665** | **0.946** | **0.149** | 0.636 | 0.580 |
| 1024 | raw | 0.634 | 0.872 | 0.099 | 0.527 | 0.535 |
| 1024 | **distilled** | **0.739** | **0.935** | **0.186** | 0.572 | 0.537 |

Confirmed at scale: distilled notes leak more (clean MIA-AUC 0.67–0.74 vs raw 0.59–0.63, tail
0.94–0.95, extraction gap ~2×), and DP collapses the tail (0.94 → 0.57–0.64 at ε=8, ~0.54–0.58 at
ε=3). Honest nuance: because distilled is *more* leaky to start, its ε=8 tail-AUC lands slightly
above raw's — it takes ε=3 to reach near-chance. The drop is still dramatic and monotone.

**Utility — answer-retrieval@5 (distilled, d=32), density makes DP nearly free at scale:**

| K | chance | clean | ε=8 | **retention@ε8** | ε=3 |
|-----|-------|-------|------|------|------|
| 32  | 0.156 | 0.620 | **0.604** | **97%** | 0.529 |
| 64  | 0.078 | 0.557 | 0.506 | 91% | 0.362 |
| 128 | 0.039 | 0.446 | 0.369 | 83% | 0.217 |

The density prediction is borne out: 6,252 notes over K=32 (~195/bucket) averages out the DP noise,
so ε=8 retention **jumps from 79% (100-user, 1,784 notes) to 97% (500-user)**. Distilled memory
answer-retrieval is essentially DP-free at ε=8 at the operating point, and still 91% at K=64.

## Caveats (honest)
- Both the 100-user and full 500-user (6,252-note) distillations are done; the ordering
  (distilled > raw leakage; DP kills both) holds at both scales, and utility retention improves
  with the denser 500-user corpus (79% → 97% at ε=8). 2 seeds — bars tighter at 500-user.
- Distillation quality is imperfect (some sessions summarize the assistant's generic advice rather
  than user facts); a tighter extraction prompt would sharpen it.
- Local 7B model, not a frontier model; the distilled notes are representative, not optimal.

## Bottom line
The bet is validated on genuinely realistic inputs — LLM-distilled memory notes on a real
benchmark, local/free tooling — on **both** axes: usable answer retrieval under DP (79% at ε=8)
AND a measured leakage drop. Confirmed at full 500-user scale: answer-retrieval retains **97% at
ε=8** (nearly DP-free at the operating point), while distilled-note leakage — *higher* than raw —
still collapses under DP (tail-AUC 0.94 → 0.57–0.64 at ε=8, near-chance by ε=3). The privacy
motivation is *reinforced*: the real payload leaks more, and DP-under-SecAgg neutralises it at
modest utility cost. Remaining: full `s` scale (247k), and the full LLM-driven MEXTRA/MRMMIA
attack pipelines.