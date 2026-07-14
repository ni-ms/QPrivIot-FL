# The paper, explained from the basics

A ground-up companion to `paper/qpriviot-memory.tex` — *"Private Federated Aggregation of
LLM-Agent Memory: Usable Shared Memory with a Measurable Leakage-Drop Guarantee under
Secure Aggregation and the Skellam Mechanism."*

This document assumes **no background** in differential privacy, federated learning, or
retrieval. Every term is defined the first time it is used, and the paper is then walked
section by section. If you already know the field, skip to [§4](#4-the-mechanism-line-by-line)
(the mechanism) or [§6](#6-the-results-what-each-number-means) (the results).

---

## Table of contents

1. [The one-paragraph version](#1-the-one-paragraph-version)
2. [The vocabulary, built up from nothing](#2-the-vocabulary-built-up-from-nothing)
3. [The problem the paper is solving](#3-the-problem-the-paper-is-solving)
4. [The mechanism, line by line](#4-the-mechanism-line-by-line)
5. [The privacy accounting, line by line](#5-the-privacy-accounting-line-by-line)
6. [The results: what each number means](#6-the-results-what-each-number-means)
7. [The three negative results (the interesting part)](#7-the-three-negative-results-the-interesting-part)
8. [What the paper does *not* claim](#8-what-the-paper-does-not-claim)
9. [Glossary](#9-glossary)
10. [Map from concept → code](#10-map-from-concept--code)

---

## 1. The one-paragraph version

An LLM agent (a chatbot with tools and persistence) keeps a **memory**: little distilled
facts about its user — *"allergic to penicillin," "prefers metric units."* That memory is
what makes the agent personal, and it is also the most sensitive thing the agent holds.
It would be useful to **pool** memory across many users, so the whole fleet of agents
benefits from what the fleet collectively knows. But you cannot just upload everyone's
notes to a server: published attacks pull verbatim user facts back out of such stores.
This paper builds a way to pool memory that (a) never sends a note anywhere in readable
form, (b) never trusts the server, and (c) carries a mathematical privacy guarantee
covering the whole pipeline — then *actually runs the published attacks against it* and
measures how much they degrade. The headline: the pooled memory keeps **95% of its
retrieval usefulness** while the strongest membership attack drops from **83% accurate to
random guessing**.

---

## 2. The vocabulary, built up from nothing

Read this section once and the rest of the paper is mechanical.

### 2.1 Embeddings — turning text into arrows

An **embedding** is a list of numbers representing a piece of text, produced by a neural
network called a **sentence encoder**. This paper uses `all-MiniLM-L6-v2`, which turns any
sentence into **384 numbers** — a point in 384-dimensional space, i.e. an arrow from the
origin.

The whole point is that *meaning becomes geometry*: sentences that mean similar things get
arrows pointing in similar directions. So you can ask "are these two sentences related?"
by asking "do these two arrows point the same way?"

- **`D` = 384** is the encoder's native dimension.
- **`d`** is the *working* dimension the paper squeezes things down to (usually 32).
- **L2 norm**, written `‖x‖₂`, is the arrow's *length*: √(x₁² + x₂² + … ). "L2" just means
  ordinary Euclidean distance.
- **Normalising** (or "L2-normalising") means rescaling the arrow to length exactly 1,
  keeping its direction. After normalising, only *direction* — meaning — survives.
- **Cosine similarity**, `cos(a, b)`, measures the angle between two arrows: **1** = same
  direction, **0** = perpendicular/unrelated, **−1** = opposite. For unit-length arrows it's
  just the dot product. **This is the paper's only retrieval operation.**

### 2.2 Retrieval — what "useful" means here

**Retrieval** = given a query, find the relevant stored item. Embed the query, embed each
stored item, return the items whose arrows point most nearly the same way as the query's.

- **top-k**: return the k best matches (this paper uses **k = 5**).
- **recall@5**: did the correct item appear in those 5? Averaged over many queries, this is
  a number between 0 and 1. The paper's variants:
  - **evidence-recall@5** — did the query reach the bucket that contains the *evidence* the
    benchmark says answers it?
  - **answer-recall@5** — same idea, on the distilled-notes corpus.
  - **topic-accuracy** — on synthetic news data, did it route to the right topic?
- **Chance**: what a random guesser scores. If you return 5 of K buckets at random, chance
  ≈ 5/K. At K = 32 that's 0.156. **Always compare a recall number to its chance line** — a
  recall of 0.10 at K = 256 (chance 0.02) is 5× chance and not at all bad.
- **Retention@ε** = `utility(private) / utility(non-private)`. "The mechanism kept 95% of
  what the clean pool could do." This isolates *the cost of privacy*, holding the
  architecture fixed. It is **not** "95% as good as having the raw notes" — see §8.

### 2.3 Random projection — shrinking 384 numbers to 32

A **projection** maps a high-dimensional arrow to a lower-dimensional one. This paper uses a
**Gaussian random projection**: build a random 384 × 32 matrix `P` (each entry drawn from a
normal distribution) and multiply. Remarkably, random directions *approximately preserve
angles*, so cosine similarity survives the squeeze. It's cheaper and, crucially here, it is
**data-independent** — the matrix depends only on a random seed, not on anyone's notes.

- **Public seed**: the random number generator's starting value is published. Anyone can
  regenerate `P`. Since it never looked at private data, **shipping it to clients leaks
  nothing** and costs zero privacy budget.
- **PCA (Principal Component Analysis)** is the *other* obvious way to reduce dimension: find
  the directions of greatest variance **in your data** and keep those. It compresses better —
  and it is **data-dependent**, which turns out to be the source of one of the paper's two
  retracted findings (§7.2).

### 2.4 LSH buckets and centroids — the shape of the shared pool

The pool cannot contain one entry per note (too many, and too identifying). Instead notes
are grouped into **K buckets**, and only a **summary of each bucket** is released.

- **LSH (Locality-Sensitive Hashing)**: a hashing scheme where *similar inputs collide on
  purpose*. The variant used is **random-hyperplane LSH**: fix K random arrows
  `a₁ … a_K` (the **anchors**, also from the public seed), and assign each note to whichever
  anchor it points most nearly toward: `bucket(i) = argmax_k ⟨eᵢ, a_k⟩`. Similar notes →
  same bucket.
- **Centroid**: the average of all the note-arrows in a bucket, renormalised to unit length.
  One arrow that "points at" the region of meaning-space that bucket covers.
- **K** is the **fidelity** of the memory. Small K (32) = few, coarse, crowded buckets. Large
  K (2048) = many, fine, sparse buckets. K is the single most important dial in the paper,
  and every table sweeps it.

**The released object is therefore K unit vectors and no text at all.** That is worth
pausing on: a receiving agent cannot read anyone's note out of the pool, because the notes
aren't in it. What the pool gives you is a **routing prior** — a map of *which regions of
memory-space the fleet knows something about* — against which any agent can route a query.
The paper is explicit that this is a *narrower promise* than "every agent inherits every fix."

### 2.5 Differential privacy — the guarantee

**Differential privacy (DP)** is a mathematical promise about a data release. Informally:

> *Whatever the released output is, it would have been almost exactly as likely to come out
> had any single user's data not been there at all.*

If that holds, no adversary — however clever, however much side information they have — can
confidently tell whether you participated. Formally, for two databases `X` and `X′` that
differ by one user, a mechanism `M` is **(ε, δ)-DP** if for every possible output set `S`:

```
Pr[M(X) ∈ S]  ≤  e^ε · Pr[M(X′) ∈ S]  +  δ
```

- **ε (epsilon)** — the **privacy budget**. Smaller = more private. ε = 0 would mean the
  output is literally independent of the data (perfectly private, perfectly useless).
  ε ≈ 1 is "strong," ε ≈ 10 is "moderate." This paper reports **ε ≈ 9.3**, and argues
  honestly about that number in §8.
- **δ (delta)** — the probability the guarantee just… fails. Must be tiny; the rule of thumb
  is `δ ≪ 1/N` where N is the number of protected units. Here **δ = 10⁻⁵** with **N = 500
  users**, so δ sits 200× below the threshold.
- **Neighbouring relation** — *what counts as "one user's data"*. This paper uses
  **add/remove one user** ("unbounded DP"): neighbours differ by whether a whole user is
  present. This is **user-level DP**, the strong version. (The weaker, more common
  *note-level* DP would protect one note at a time and is much cheaper — the paper is
  careful never to sneak note-level reasoning into a user-level claim.)
- **Sensitivity (Δ₂)** — how much the *un-noised* answer can move if one user is added or
  removed. This is the quantity the noise must be calibrated to. Bigger sensitivity → more
  noise → worse utility. Everything in §4 about clipping exists to make Δ₂ small and,
  more importantly, **exactly known**.
- **Post-processing immunity** — a core DP property: once output is (ε,δ)-DP, *anything* you
  compute from it is still (ε,δ)-DP. You can never leak more by thinking harder about a
  private release. (This is why normalising the noisy sum is free, and why the "free"
  occupancy rule in §7.3 costs nothing.)
- **Composition** — release the same data twice and the ε's **add up**. Two releases at
  ε = 9.3 is a ε = 18.6 mechanism, not a ε = 9.3 one. This is the whole story of §5.4.

### 2.6 The noise: Gaussian, discrete Gaussian, Skellam

DP is achieved by **adding random noise** to the answer. The classic choice is **Gaussian**
(bell-curve) noise, scaled to the sensitivity.

Two problems in a federated setting:

1. Gaussian noise is a **real number**. Cryptography works on **integers**, in finite
   arithmetic (mod p). A float doesn't survive.
2. The noise has to be added *without a trusted party*, so each of the N users adds a
   **share** of it — and the shares must **add up to** the intended total noise.

The **Skellam distribution** solves both. It is the **difference of two Poisson random
variables** (a Poisson counts random events; the difference of two can be negative, which is
what you need for noise). It is:

- **discrete** — integer-valued, so it survives the crypto path; and
- **closed under addition** — *the sum of independent Skellams is itself a Skellam.* This
  is the magic property. It means N users each adding `Skellam(μ/N)` produces, in the sum,
  exactly `Skellam(μ)`. The distributed release **equals** the intended central one.

A **continuous Gaussian does not have this property under modular arithmetic**, which is
precisely why the nearest prior work (a centralized DP datastore) cannot be lifted to this
federated setting for free.

- **σ (sigma)** is the noise scale knob. Bigger σ = more noise = smaller ε = worse utility.
  The paper's main operating point is **σ = 0.606 → ε ≈ 9.3**.
- **Quantisation** is the step that turns real-valued vectors into the integers Skellam and
  the crypto need: multiply by a scale factor `s` and round. `range_max = 10⁶` is the grid
  resolution; §6.4 shows this rounding costs **essentially zero privacy** (a surcharge of
  1.3 × 10⁻⁹ on ε).

### 2.7 Secure Aggregation (SecAgg) — the server never sees a summand

**SecAgg** is a cryptographic protocol that lets a server compute `Σᵤ xᵤ` — the *sum* of all
users' vectors — **without ever seeing any individual `xᵤ`**.

The trick is **zero-sum masks**: each pair of users agrees (via key exchange) on a shared
random vector; one adds it, the other subtracts it. Every user's submission is therefore
their real data plus a big pile of random garbage — individually meaningless. But because
the masks are constructed to sum to zero, **when the server adds everything up, all the
masks cancel** and the true sum falls out.

So: the server gets the sum, and nothing else. **No trusted curator.** This is the "curator-free"
claim in the title.

- **Client dropout**: SecAgg's classic failure mode. If some users go offline mid-protocol,
  their masks don't cancel and (in this paper's concern) their *noise shares* are missing
  too — so the release ends up with **less noise than promised** and the ε you're quoting is
  a lie. §5.3 handles this.
- **`M_min`**: the paper's fix — a **survival floor**. Size each client's noise share to
  `μ/M_min` rather than `μ/N`, so any turnout above the floor produces **at least** the
  target noise. The guarantee can only improve, never silently degrade. If fewer than
  `M_min` clients show up, the server **aborts** instead of releasing.

### 2.8 The attacks — how privacy is *measured*, not just claimed

A DP proof says "an attack cannot succeed." The paper insists on also *running the attacks*.

**Membership inference attack (MIA)** — the fundamental privacy attack. Given a candidate
note and the released pool, decide: **was this person's data in the pool?** That's it. It
sounds mild until you notice the pool might be "users who asked about a medication."

Scored with:

- **ROC-AUC** (Area Under the Receiver Operating Characteristic curve). Sweep the attack's
  decision threshold and plot true-positive rate against false-positive rate; AUC is the area
  under that curve. **0.5 = the attack is a coin flip (no leakage). 1.0 = perfect
  re-identification.**
- **TPR@1%FPR** — "true-positive rate at a 1% false-positive budget." *Of the members, what
  fraction can the attacker confidently identify while wrongly accusing only 1% of
  non-members?* This is the **operationally meaningful** metric — an attacker in the real
  world cares about confident hits, not average-case ranking — and AUC-only evaluations hide
  it. **The floor is 0.01** (at a 1% FPR you get 1% TPR by guessing). The paper headlines
  this metric.

Two attack *strengths*, and the difference between them is a whole finding:

- **The uncalibrated cosine proxy** (the weak one). Score a candidate by
  `cos(x, μ[bucket(x)])` — how close is this note to its own bucket's centroid? Members
  pulled that centroid toward themselves, so they should score higher. This is the
  measurement principle of the published agent-memory attacks, and it is the natural first
  thing to reach for. **It is also nearly useless at coarse K**, for reasons in §7.1.
- **LiRA (Likelihood Ratio Attack)** (the strong one, the modern MIA standard). Instead of
  comparing a candidate's score against the *population*, compare it against **itself**:
  build many **shadow releases** — full re-runs of the aggregation, some *with* this exact
  note included, some *without* — and learn the two score distributions for **this specific
  target**. Then the test is a per-target **likelihood ratio**: is the observed score more
  likely under the "in" distribution or the "out" distribution?
  - This is normally expensive (each shadow release means retraining a model). Here
    aggregation is **numpy**, not training — so 48 shadow releases take **34 seconds**. The
    paper leans hard on this: the strong attack is *cheap* in this setting, so there is no
    excuse for the weak one.
- **Extraction / reconstruction** — not "was this person in?" but "**what did they say?**"
  - **Decode-extract** (the paper's version): take a released centroid, find the nearest note
    to it from a candidate set, and score a hit if that note really is in that bucket. This is
    a **closed-set** attack — the adversary is *handed the candidate notes* and only has to
    pick. §7.3 explains why this metric turns out to be the wrong instrument.
  - **Open-set embedding inversion** (e.g. `vec2text`) — recover the note's *text* from the
    vector, with no candidate list. **The paper does not run this, and says so loudly.** It
    is the acknowledged chief gap.

**The two published attacks the paper reuses as its harness** (it doesn't claim to have
invented them):

- **MEXTRA** — a prompt-injection attack that makes a live agent *recite its raw-text
  memory verbatim*.
- **MRMMIA** — membership inference against a live agent's memory.

---

## 3. The problem the paper is solving

### 3.1 Why pool memory at all

Frameworks like **A-MEM** and **Mem0** give each user's agent a private store of distilled
notes. Sharing across users would let a fleet profit from what the fleet knows. But raw
pooling is a disaster, and *concretely* so: MEXTRA and MRMMIA are published, working attacks
against exactly this artifact.

### 3.2 Why the obvious defenses don't suffice

- **Access control** (*Collaborative Memory*) — governs *who reads* a note. No guarantee
  about the **aggregate**.
- **On-device redaction/masking** (*MemPrivacy*) — hides *what's in* a note. Again, no
  aggregate guarantee.
- **A curated central DP datastore** (*DP Datastore Generation*) — has a real DP guarantee,
  but reintroduces a **trusted curator** who sees everything, and uses continuous noise with
  no secure aggregation.

The gap: a mechanism that is **(a)** curator-free, **(b)** gives a **central-DP guarantee over
the pool**, and **(c)** is *shown* to degrade the known attacks while remaining useful.

### 3.3 The threat model, stated precisely

- **Parties**: N users with local agents; an **honest-but-curious** server (it follows the
  protocol but will happily analyse everything it sees); downstream agents that query the
  pool.
- **Adversary**: a **recipient of the released pool** — the server itself, or any agent that
  can query it. **Passive.**
- **What is out of scope, explicitly**:
  - **Active, colluding clients.** A malicious participant could inject crafted notes to
    steer particular centroids, and query before/after a target joins — turning the offline
    LiRA into an *adaptive online* attack. Robust aggregation against poisoning is an open
    problem, orthogonal to the release mechanism.
  - **Participation metadata.** SecAgg hides *contents*, not the *fact* that you participated.
    The paper specifies mandatory participation with padded dummy payloads to close this —
    and states plainly that it **specifies but does not measure** it.

---

## 4. The mechanism, line by line

This is Algorithm 1 of the paper, in English. The **public parameters** — known to everyone,
including the adversary, and costing **zero privacy** because none of them looked at private
data — are: the LSH anchors `{a₁…a_K}`, the projection `P`, the clip bound `C`, the quantiser
bound `B`, the noise scale `σ`, the survival floor `M_min`, and the clip-selection budget `ε_c`.

### Server, once (setup)

**1. Choose the clip bound `C` under DP.**
This is subtle and it's one of the paper's fixes to its own earlier draft. `C` is the maximum
allowed length of a user's payload — everything longer gets shrunk. The *natural* choice is
"the 95th percentile of the observed payload lengths"… but that is **a function of the private
data**, and shipping it to every client is therefore **an unaccounted release**. You'd be
leaking, and not counting it.

So `C` is instead selected with the **exponential mechanism** — a standard DP primitive for
*choosing* something privately: score every option on a public grid, then sample an option with
probability proportional to `exp(ε_c · score / 2)`. Good options are likely; bad ones aren't
impossible. Cost: **ε_c = 0.1, which is 0.2% of the total budget.**

The selection is deliberately **conservative** — it lands ~1.6× above the true p95, which means
it **over-noises**. That can never *under*-protect, so the guarantee stays safe; the price is a
2% dent in utility (evidence-recall 0.416 → 0.406). The paper takes that trade and says so.

**2. Set `B := C`, and derive the quantiser step `s = range_max / B`, and the target noise
variance `μ = (σ·C·s)²`.**
`B` (the quantiser bound) is **free**: since each note is unit-length and the payload is
clipped *first*, no single coordinate can exceed the whole vector's length `C`. So `B := C` is
a *provable public bound* that never clips anything — and ε doesn't even depend on `B`, because
`Δ₂/√μ = 1/σ` regardless.

### Client `u` (on device — **note text never leaves**)

**3–5. Embed, project, bucket.**
Each note → sentence encoder → 384-d vector → project down to `d` → normalise to unit length →
assign to the bucket whose anchor it points most nearly toward. Because `P` and the anchors come
from a **public seed shared by everyone**, all users' notes land in a **common coordinate frame**
and comparable buckets. This step is data-independent and therefore free.

**6. Form the per-bucket sum-vector.** `v_u[k] = Σ (notes of user u in bucket k)`. Just add the
arrows up.

**7. Concatenate into the full payload.** `V_u = [v_u[1]; …; v_u[K]]`, a vector of `K·d` numbers.
This — *the whole thing* — is what the user is releasing, and so this is what the sensitivity
must be computed over.

**8. Clip GLOBALLY.** `V_u ← V_u · min(1, C/‖V_u‖₂)`. If the payload is longer than `C`, shrink
it to exactly `C`; otherwise leave it.

> **Why "globally" is the whole ballgame** (Remark 1 in the paper). The tempting alternative is
> to clip **each bucket vector separately** to `C`. That sounds equivalent. It is not. A user with
> notes in `b_u` distinct buckets would then contribute up to `C` **in each one**, so their true
> payload sensitivity is `√b_u · C`, not `C`. An accountant still charging `Δ₂ = C·s` would
> **under-count by a factor of √b_u** — and the paper measured what that means here: the busiest
> user touches 21 buckets at K = 32 and 54 at K = 1024, which would have inflated the real budget
> from the reported **ε ≈ 9.3 to ε ≈ 69, or ε ≈ 159**. That is a broken guarantee, silently.
> Clipping once, on the concatenated payload, is what makes `Δ₂ = C·s` **exact**.

Note also that the clip is a **scalar rescale**: it shrinks a heavy user's magnitude but leaves
the *direction* of `V_u` untouched. A prolific user is **down-weighted, never distorted**.

**9–12. Quantise, noise, mask, submit.** Per bucket: multiply by `s` and round to integers; add
this client's **share** of the Skellam noise, `Skellam(μ / M_min)`; add the SecAgg zero-sum mask;
submit.

Order matters here. The noise is injected **before** masking, which is why an idle client (all-zero
payload + noise + mask) is **distributionally indistinguishable** from an active one. Neither the
server nor a network observer learns who actually used their agent this epoch.

### Server (honest-but-curious — sees only masked sums)

**13. Abort if fewer than `M_min` clients survived.**

**14–15. Sum, and normalise every bucket.** The masks cancel; the server is left with
`Σ z_u[k] + Skellam((M/M_min)·μ)` — the true sum plus at-least-the-target noise. Divide by `s`,
normalise to unit length. That is the private centroid.

> **"Every bucket" includes the empty ones, deliberately.** An empty bucket contains *pure noise*,
> and normalising pure noise manufactures a random unit vector that will compete for top-5 slots
> against genuine centroids. The obvious fix — suppress buckets whose true count is zero — requires
> **reading the true counts**, which is a non-private read of the corpus and *exactly* the
> unaccounted-release error the paper indicts elsewhere. So the paper **refuses to do it** and pays
> the cost in utility instead. (At K ≤ 256, where every reported retrieval number lives, no bucket is
> ever empty, so the issue is moot; see §7.3.)

**16. Release the K centroids.** No text. No counts.

### Retrieval (any downstream agent)

**17.** Embed the query with the same public encoder + projection, and return the top-5 buckets by
cosine similarity to the released centroids. Because DP is immune to post-processing, **nothing an
agent does with the pool can leak more than the pool already did.**

### Why only the vectors, and not the counts (§5.2 of the paper)

The obvious design releases **two** channels: the sum-vectors *and* the per-bucket counts (so you
can compute a true average). That costs **two compositions** — roughly double the ε.

But the count is **redundant for retrieval**: `normalize(sum / count) = normalize(sum)`. Dividing
by a scalar and then renormalising *cancels the scalar exactly*. The count magnitude cannot change
the ranking. So the paper releases **the vector channel only**: **identical retrieval, to the last
decimal, at ~1.5× tighter ε (14.0 → 9.3).**

The one thing the count channel *was* silently supplying is **occupancy** — *which buckets are
non-empty*. That does **not** cancel. §7.3 prices it and shows it cannot pay for itself.

---

## 5. The privacy accounting, line by line

### 5.1 The ε formula

Converting "how much noise did I add" into "what ε is that" uses **Rényi Differential Privacy
(RDP)** — a reformulation of DP indexed by a parameter α that **composes additively** and converts
cleanly to (ε, δ). The paper uses the **exact Skellam-RDP bound** (Agarwal, Kairouz & Liu, NeurIPS
2021, Thm 3.5):

```
ε_RDP(α) ≤ α·Δ₂² / (2μ)  +  min{ ((2α−1)·Δ₂² + 6·Δ₁) / (4μ²) ,  3·Δ₁ / (2μ) }
```

- `μ` — the per-coordinate released noise **variance**.
- `Δ₂ = C·s` — the L2 sensitivity (exact, *because* of the global clip).
- `Δ₁ ≤ √d · Δ₂` — the L1 sensitivity (sum of absolute coordinates rather than Euclidean length).
  The Skellam bound needs it; the Gaussian one doesn't. This is the (small) price of going discrete.
- The first term is the familiar Gaussian-like `α·Δ₂²/2μ`. The `min{…}` is the **discretisation
  surcharge** — and at `range_max = 10⁶` it is **1.3 × 10⁻⁹**, i.e. **discretisation is free**.

Then convert to a single (ε, δ) by minimising over α:
`ε = min_α [ RDP(α) + ln(1/δ)/(α−1) ]`, with δ = 10⁻⁵.

The clip-selection cost folds in via **pure-DP → zCDP conversion**: an ε_c-DP mechanism is
(ε_c²/2)-zCDP, hence `RDP_α ≤ α·ε_c²/2`, which just adds to the sum.

### 5.2 The σ → ε map (the number to memorise)

| σ | vector-only ε (recommended) | two-channel ε |
|---|---|---|
| 0.303 | 22.11 | 33.31 |
| **0.606** | **9.30** | 13.94 |
| 1.615 | **3.21** | 4.63 |
| 2.854 | **1.81** | 2.56 |

**σ = 0.606 → ε ≈ 9.3** is the paper's operating point throughout. (The codebase's exploration
labels "ε = 8 / ε = 3" were an older, approximate Gaussian labelling; the rigorous values are 9.3
and 3.2. Since every effect is **monotone in σ**, relabelling the axis changes no ordering, no
retention percentage, and no AUC.)

### 5.3 Client dropout: the noise must be sized to *survivors*

If each of N clients naively injects `Skellam(μ/N)` and only M survive, the server recovers only
`(M/N)·μ` of variance — the effective noise is `σ·√(M/N)`, and **the guarantee degrades silently,
worst exactly when turnout is worst**:

| survivors M/N | 1.00 | 0.90 | 0.70 | 0.50 | 0.25 |
|---|---|---|---|---|---|
| **true ε** (nominal 9.30) | 9.30 | 9.91 | 11.61 | **13.94** | **22.11** |

Two sound fixes: **(a)** the `M_min` survival floor (each client injects `μ/M_min`, so any turnout
above the floor can only *over*-noise; abort below it), or **(b)** fault-tolerant distributed noise
generation (secret-share the noise seeds, exactly as SecAgg already does for masks, so survivors can
reconstruct the dropped clients' shares).

The paper's experiments assume **full participation** and it records dropout robustness as
**specified but not measured** — a limitation, not a claim.

### 5.4 Repeated release: a memory is not a model

**A model is trained once. A memory is re-aggregated forever.** Users keep writing notes, so the
release happens **T times**, and RDP composes **additively**. Re-releasing at the single-shot
σ = 0.606 costs:

| T | cadence | σ needed to hold ε = 9.31 | recall@5 | retention | **naive ε if you don't re-tune** |
|---|---|---|---|---|---|
| 1 | single shot | 0.606 | 0.406 | **95%** | 9.3 |
| 4 | quarterly, 1 yr | 1.211 | 0.382 | 89% | 22.1 |
| **12** | **monthly, 1 yr** | 2.098 | 0.349 | **81%** | 44.2 |
| 30 | daily, 1 month | 3.317 | 0.313 | 73% | **93.3** |
| 52 | weekly, 1 yr | 4.367 | 0.283 | 66% | **153.3** |

Read the last column first: **a mechanism advertised at ε ≈ 9.3 and re-run nightly is not an
ε ≈ 9.3 mechanism — it is an ε ≈ 93 mechanism.** This term dwarfs every other in the accounting by
two orders of magnitude, and it is the one most easily left implicit.

But it is **payable**. Composed RDP is linear in T while per-release RDP falls as 1/σ², so the noise
needed to hold a **fixed total budget** grows only as **√T**. You can simply *buy* a cadence up
front: pick your T, solve for σ, pay once in utility. **Monthly re-aggregation for a full year fits
inside the same ε ≈ 9.3, at 81% retention.**

Two things make this cheaper than it looks: the **public parameters don't recompose** (the projection
and anchors are public-seed → zero ε at any T; `C` is a *public bound*, selected once and reused —
charged **once**, not T times), and the guarantee is **user-level and unbounded**, so a user who
writes nothing between rounds submits an all-zero payload and **their privacy is not re-spent by the
calendar**. It is the *release* that composes, not the user.

---

## 6. The results: what each number means

### 6.1 The experimental setup

- **LongMemEval (oracle)** — real long-horizon conversational memory. Users = question haystacks,
  notes = dialogue turns, queries = questions, ground truth = the evidence turns. **500 users,
  10,957 notes, 479 queries.**
- **LongMemEval (`s`)** — same 500 users, **246,073 notes** (~96% distractors), ~22× bigger. Used
  for the at-scale check.
- **LLM-distilled notes** — dialogue turns distilled into A-MEM/Mem0-style memory notes by a local
  LLM (Ollama `qwen2.5:7b`). **500 users → 6,116 notes.** *This is the realistic agent-memory
  payload*, and the store for the live-agent attacks.
- **20-Newsgroups controls** — for clean, controlled N/K/d sweeps.
- **Protocol**: 5 seeds, **final (not peak) metric**, mean ± std. LiRA is 3 seeds × 48 shadow
  releases. Members vs non-members are a same-distribution split (no train/test confound).

### 6.2 The joint frontier — the paper's central table

This is the discipline the whole paper is organised around: **a privacy/utility claim is a claim
about ONE release, so both axes must be measured at the SAME operating point.** (The paper's own
superseded draft failed this: it reported utility only at K ≤ 256 and leakage only at K ≥ 512 — two
disjoint grids — and then quoted a headline from each as if they described one mechanism. They
didn't.)

| K | chance | clean recall | recall @ ε 9.3 | **retention** | LiRA AUC clean | LiRA AUC @ ε | TPR@1%FPR clean | TPR@1%FPR @ ε |
|---|---|---|---|---|---|---|---|---|
| **32** | 0.156 | 0.428 | **0.406** | **95%** | **0.833** | **0.502** | 0.175 | **0.011** |
| 64 | 0.078 | 0.320 | 0.276 | 86% | 0.908 | 0.505 | 0.357 | 0.011 |
| 128 | 0.039 | 0.233 | 0.167 | 72% | 0.954 | 0.502 | 0.552 | 0.011 |
| 256 | 0.020 | 0.167 | 0.100 | 60% | 0.981 | 0.503 | 0.750 | 0.011 |
| 512 | 0.010 | 0.152 | 0.062 | 41% | 0.993 | 0.505 | 0.874 | 0.013 |
| 1024 | 0.005 | 0.135 | 0.032 | 24% | 0.997 | 0.504 | 0.945 | 0.013 |
| 2048 | 0.002 | 0.128 | 0.016 | 13% | 0.998 | 0.505 | 0.957 | 0.012 |

**Three readings, all load-bearing:**

1. **DP defeats the calibrated attack at *every* K.** LiRA AUC lands in 0.502–0.505 and TPR@1%FPR at
   the 0.011–0.013 false-positive floor — *for all seven fidelities* — even as the **clean** attack
   climbs to a near-perfect 0.998. **The privacy axis is FLAT in K.** It does not trade against
   fidelity at all.
2. **Therefore K is a pure utility decision, and coarse wins.** Retention falls monotonically
   95 → 86 → 72 → 60 → 41 → 24 → 13% as K grows. Why: utility is governed by **effective contributors
   per bucket**. Coarse buckets pool many users, so the signal is loud and the fixed noise barely
   dents it. Fine buckets **starve** — few notes each — so the same noise swamps them.
   **K = 32 is the recommendation: 95% retention with the membership attack at the floor.**
3. **Membership is defeated everywhere; closed-set reconstruction is not.** Decode-extract at K = 32
   only halves (0.771 → 0.396) against a *measured* chance floor of 0.010 — i.e. **40× above chance**.
   The paper states this plainly rather than rounding it away, and then argues the metric is the wrong
   instrument (§7.3).

### 6.3 Utility in detail, and the honest caveat

At K = 32, d = 32: **95% retention at ε ≈ 9.3, 85% at ε ≈ 3.2**, at 2.6× chance.

**What "95%" does and does not mean.** Retention is `utility(ε) / utility(clean pool)` — it isolates
**the cost of the privacy mechanism**, holding the architecture fixed. It is **not** the cost of the
architecture. Bucketing into K centroids is *itself* a lossy retriever: on the controls, a plain
un-bucketed nearest-neighbour search over the same notes scores 0.43–0.65 where the clean bucketed
pool scores 0.19–0.32.

> **Coarse bucketing costs roughly half the retrieval utility, and DP then costs a further 5%.**

The paper states this explicitly because "95%" is otherwise easy to misread as "the private pool is
95% as good as having the notes." It isn't. But that un-bucketed retriever is **not an available
baseline** — it requires every note in the clear at query time, which is precisely the artifact you
may not release (and precisely what MEXTRA eats). It's an upper bound, not a competitor. Honest
framing: *a releasable shared pool costs about half of an unreleasable one, and making it
differentially private on top of that costs 5%.*

### 6.4 Where the ε actually goes

| term | ε | note |
|---|---|---|
| Continuous-Gaussian RDP reference | 9.2909 | reference point |
| + discretisation (Skellam) | 9.2909 | surcharge 1.3 × 10⁻⁹ — **free** |
| + projection (public seed) | 9.2909 | data-independent → **+0.000** |
| + DP clip-bound selection (ε_c = 0.1) | **9.3109** | **+0.020 (0.2%)** |

Three of the four terms are zero or negligible. **The reported budget covers the whole pipeline, not
just its last stage** — and the paper argues (§8) that the smaller ε's commonly reported for pipelines
of this shape typically account only for the final noise addition and silently exclude a
data-dependent preprocessing step of exactly the kind §7.2 shows to be both leaky *and* distorting.
The numbers are not comparable in the direction a reader would assume.

### 6.5 At scale (246k notes): DP becomes free

On the `s` variant, retention is **97–100%** across K = 32…256 — the clean/private differences (≤ 0.003)
are an order of magnitude *below* the 5-seed std. **At this density DP costs nothing measurable.**
(The 100–101% cells are not evidence that noise helps; they're noise.)

### 6.6 The realistic payload: LLM-distilled notes

The payload a production agent actually stores. Utility: **89% retention at ε ≈ 9.3** at 500 users
(the 100-user pilot only gets 65% — **density, not scale per se, governs retention**). And the
calibrated attack tells the same story as on raw turns:

| payload | release | AUC | TPR@1%FPR | decode |
|---|---|---|---|---|
| raw turns | clean | 0.833 | 0.175 | 0.771 |
| raw turns | ε ≈ 9.3 | 0.502 | **0.011** | 0.396 |
| **distilled notes** | clean | **0.838** | **0.171** | 0.750 |
| **distilled notes** | ε ≈ 9.3 | 0.503 | **0.011** | 0.302 |

**The endpoint holds on the object a production agent stores, at the same operating point and the
same budget.** (This table also *retracts* an earlier, more dramatic reading — see §7.1.)

### 6.7 The published attacks, against a live agent

Build an actual A-MEM/Mem0-style agent (local `qwen2.5:7b`) that retrieves shared memory into its
context and answers. Run **MEXTRA** and **MRMMIA** end to end. Hold the notes fixed; swap only the
shared-memory back-end:

| back-end | MEXTRA verbatim recovery | MRMMIA AUC |
|---|---|---|
| **raw text** (status quo, no privacy) | **1.000** | **0.981** |
| **our SecAgg+Skellam pool** (ε ≈ 9.3) | **0.017** | 0.526 |

On a raw-text memory both attacks succeed **decisively**: every targeted note is reproduced verbatim
and membership is inferred at AUC 0.98.

Two pieces of honesty in the paper here, and they're worth copying:

- **The 0.017 is not zero, and the paper refuses to round it.** Verbatim recovery from a centroid pool
  is *structurally impossible* — there is no text in it. So 1.7% is the **false-positive rate of the
  recovery detector**: a decoded *public* note that happens to sit within the 0.9 similarity threshold
  of a target. **It measures our measurement, not the release.**
- **A harness bug, reported rather than buried.** The membership probe originally drew non-members
  from the same held-out set that served as the adversary's public decode corpus — so in private mode
  the non-members appeared *verbatim in the agent's context* while members never could, and the attack
  scored **below chance** (AUC 0.31–0.35). *Any AUC reliably below 0.5 is a design error, not a privacy
  result.* The corpus is now split three ways, pairwise disjoint.

And the correct claim, stated as such: **MEXTRA is not "defeated," it is structurally immunised** —
the attack's target object no longer exists. Any vector-only datastore inherits that for free, so
claiming victory over it would be a strawman. This section's role is to show what the **status quo**
leaks, not to certify the release.

---

## 7. The three negative results (the interesting part)

The paper's most transferable content is three findings that **overturned results it had already
written up**. Each would have been caught by **one control run**.

### 7.1 The weak attack is blind exactly where you would deploy

Compare the two attacks **on the identical clean release** at K = 32:

- the **uncalibrated cosine proxy** reads **AUC 0.504** — chance to three decimals. On this evidence
  there is no privacy problem to solve.
- the **calibrated LiRA**, on that *same* release, reads **AUC 0.833, TPR@1%FPR 0.175 (17× the floor),
  and reconstructs 77% of centroids to a true in-bucket note.**

**The pool was never safe. The proxy simply could not see it.**

*Why:* an uncalibrated cosine score compares a member's similarity against a **population** baseline.
At coarse K the population baseline is *high* — every candidate is close to a centroid that averages
hundreds of notes. Only a **per-target likelihood ratio**, calibrated against shadow releases with and
without that exact note, can separate *"close because the bucket is broad"* from *"close because I am
in it."*

Worse, it **manufactures a false story about where leakage lives.** The proxy's sensitivity *grows*
with K (0.504 → 0.738 as K goes 32 → 2048) while the calibrated attack is already strong at K = 32. So
a proxy-only evaluation concludes that leakage is a **sparse, high-K, low-count-tail curiosity** —
present only in a regime whose 13–41% retention nobody would ever ship. That is exactly what the
superseded draft concluded, and it is wrong twice: it understates the threat at the deployed operating
point, *and* it invites the reader to dismiss the mechanism as solving a problem that only exists where
the system is useless anyway.

> **An evaluation that reports only an uncalibrated proxy has not measured privacy. It has measured its
> own attack.**

The remedy is cheap: LiRA here needs no model training, and 48 shadow releases run in **34 seconds**.

*(The same lens also explains the retraction in §6.6: the proxy had reported that distilled notes leak
**more** than raw turns — "distillation concentrates identifying facts," AUC 0.68 → 0.80. Under
calibration that gradient **does not survive** (0.838 vs 0.833). The paper drops the more dramatic
reading and keeps the true one.)*

### 7.2 The dimension "crossover" was an artifact of data-dependent preprocessing

An earlier draft reported a **dimension/density crossover**: growing `d` appeared to *simultaneously*
raise pre-DP leakage and lower DP utility-retention — making tiny-`d` a **joint optimum for privacy and
utility at once**. A free lunch. A great finding.

It was an artifact of the **projection**, which had been a **PCA fit on the users' own notes**.

Why that's fatal, twice over:

1. **It is an unaccounted release.** `P` is handed to every client as a "public parameter" while being
   a **function of private data**. The reported ε covered only the second stage. You were leaking and
   not counting it.
2. **It fabricates the finding.** PCA orders directions by variance **in the users' data**. At small `d`
   it keeps exactly the directions the corpus **shares** and discards the **idiosyncratic tail**. That
   simultaneously *flatters clean retrieval* **and** *destroys the individuating signal an attacker
   needs* — which is precisely the "both axes at once" effect. **But it purchases both with the same
   illicit currency: information about the private corpus, spent outside the accountant.** A random
   projection, spending nothing, buys neither.

The control that catches it: **at d = 384 the projection IS the identity**, so all three projection
choices *must* coincide there. Re-running the whole sweep under `data-PCA`, `randproj` (public-seed
Gaussian) and `publicPCA` (PCA fit on a *disjoint public* corpus), holding everything else fixed:

| d | data-PCA (leaky) | randproj (free) | publicPCA (free) |
|---|---|---|---|
| **32** | **0.308** | 0.151 | 0.147 |
| 64 | 0.285 | 0.178 | 0.150 |
| 128 | 0.240 | 0.158 | 0.142 |
| **384 (identity — the control)** | **0.161** | **0.161** | **0.161** |

Tiny-`d` wins by **91%** under data-PCA (0.308 vs 0.161 at the identity). Under either honest
projection **the advantage disappears into the seed noise** (spreads of 0.03 against a 5-seed std of
±0.02–0.03). And on the leakage axis, the gradient that motivated the claim (0.944 → 0.989) flattens
to 0.988 → 0.989.

The paper is careful not to over-claim in the other direction: the honest statement is **not** that
tiny-`d` becomes the *worst* choice (no ordering among `d` is statistically resolvable at 5 seeds) but
that **the effect vanishes.** *What replaces the crossover is a flat line, not a reversal.*

**What survives:** tiny-`d` is still a fine **engineering** default — at d = 32 the payload is **12×
smaller** (1,024 vs 12,288 coordinates) for no measurable privacy or utility cost. That's a
communication argument, not a free lunch.

**And the generalisable warning:** *data-dependent preprocessing (PCA, whitening, vocabulary selection,
learned quantisers, clip bounds read off the data) is both an unaccounted release AND a confound that
can manufacture an effect out of nothing. It is standard practice. An identity-projection control costs
one extra run and would catch it.*

### 7.3 Two quantities that cannot be separated from utility

The third result is really a *shape* that shows up twice.

**(a) Closed-set reconstruction is the same signal as retrieval.** Decode-extract at K = 32 is 0.396
against a chance floor of 0.010 — DP does *not* defeat it. But look at what the metric is made of:

| K | decode @ ε 9.3 | retention |
|---|---|---|
| 32 | 0.396 | 95% |
| 128 | 0.070 | 72% |
| 2048 | 0.001 | 13% |

**Decode falls exactly as retention falls.** That is not a coincidence, it is a **tautology**: a centroid
is useful for retrieval *precisely when* it points at the notes in its own bucket. "The nearest note to
`μ̃[k]` lies in bucket k" is a **restatement of** "the pool retrieves." **You cannot drive closed-set
decode to chance without destroying the pool** — the only row that reaches the floor (K = 2048) is the
row whose 13% retention makes it worthless.

And a closed-set "hit" **discloses nothing the adversary didn't already have**: the attack *hands them
the true candidate notes*. All a hit certifies is that a note they **already hold** falls in bucket k —
and `bucket(x) = argmax_k ⟨x, a_k⟩` is computable **by anyone, from the public anchors, with no access
to the release at all.** The metric is an **upper bound** on inversion, and upper bounds are informative
in one direction only: the low values at K ≥ 1024 *do* certify nothing is recoverable there; the high
value at K = 32 certifies **nothing at all**.

**(b) Occupancy and membership are the same signal.** An empty bucket normalises to a random unit vector
that competes for top-5 slots. Can you privately detect and suppress empty buckets?

- **At the operating points the question is moot**: with 10,957 notes over 500 users, **no bucket is
  empty for K ≤ 128 on any seed**, and 0.23% at K = 256. Every retrieval number in the paper is at
  K ≤ 256. The failure mode is **absent from the reported utility, not tuned away.**
- **Where it does arise, it cannot be bought cheaply.** Under **user-level** DP a count channel is *not*
  the cheap sensitivity-1 object it would be under note-level DP — deleting a user erases **all** its
  entries at once, so even a **binary user-indicator** has L2 sensitivity `√b_u` (≈ 5–8 here), **not 1**.
  The **free** rule (threshold the released vector's own norm against the analytic noise floor — free by
  post-processing) is **at chance** (AUC 0.48–0.50). The accurate rule costs a **full second composition**
  (ε 9.31 → 13.94).
- **And that is intrinsic, not an implementation shortfall**: *the noise that drives membership inference
  to chance in a one-contributor bucket is the very same noise that hides whether the bucket has a
  contributor at all.* **No mechanism hides the member and reveals the bucket.**

The design consequence in both cases: **operate at dense, coarse K** — where retention is highest
anyway — rather than bolt on a channel that cannot pay for itself.

> The moral shared by (a) and (b): **a quantity inseparable from utility cannot be independently
> suppressed.** When a "privacy metric" moves in lock-step with your utility metric, it is not measuring
> privacy.

---

## 8. What the paper does *not* claim

The paper is unusually disciplined about this, and reading these limits is the fastest way to understand
what it *does* claim.

- **It does not claim to defeat extraction.** **Open-set embedding inversion** (recovering a note's
  *text* from a released centroid, `vec2text`-style, without being handed the candidates) is **not run**.
  It is named as **the paper's chief gap** and "the single most important experiment left undone."
  *A reader who needs reconstruction resistance today, rather than membership resistance, should not treat
  this paper as having supplied it.*
- **It does not claim the pool transfers knowledge.** The release is K vectors and **no text**. It is a
  **routing prior** — a map of which regions of memory-space hold knowledge — **not** a note exchange.
  Utility is measured as *routing fidelity*, not answer quality. And the paper probed the downstream
  question and **reports that it does not close**: on LongMemEval, an agent answering with the private
  pool in context does **no better than with no pool at all** — and even a *leaky raw-text* pool helps only
  marginally. The reason is structural: **LongMemEval is a personal-recall benchmark** ("what was the first
  issue *I* had with my car"), where every answer lives in **one user's own history**, so a *cross-user* pool
  has nothing relevant to contribute. It is the wrong benchmark for a shared knowledge prior — and it is
  the only realistic agent-memory corpus with ground-truth queries currently available. Building a benchmark
  where knowledge genuinely **transfers** across users is the biggest piece of future work this paper points
  to.
- **It does not claim ε ≈ 9.3 is a tight budget.** It isn't, by DP-literature standards, and the paper says
  so. Two things are offered in its defence: it is an **end-to-end** budget (the projection, the anchors, the
  clip bound and the quantiser bound are *all* public or DP-selected), whereas smaller published ε's for
  pipelines of this shape typically cover only the last noise-addition stage; and **the empirical endpoint is
  what the budget is for** — at this ε *every* attack the authors can mount is driven to its floor.
- **It does not claim novelty for the crypto.** SecAgg + Skellam is prior work, used **verbatim**. The
  LSH-bucketing-plus-DP skeleton overlaps a centralized prior datastore. **The novelty is the payload**
  (agent-memory embeddings), **the federated curator-free composition**, **the joint frontier with a measured
  attack-drop endpoint**, and **the two methodological refutations**.
- **Specified but not measured**: client dropout robustness; participation-metadata padding; topic drift
  across a multi-round deployment; δ at production scale (a 10⁵-user fleet must retighten δ — cost: **< 23% in
  ε** for three orders of magnitude).
- **Explicitly out of scope**: active/colluding clients (poisoning the centroids), and a stronger adaptive
  online attack built on that.

**What it *does* claim, precisely:** *membership inference — the threat this mechanism is designed against,
and the one the published attacks mount — is driven to the false-positive floor at every fidelity measured,
while retrieval keeps 95% of its clean utility at the recommended operating point, under a budget that covers
the entire pipeline.*

---

## 9. Glossary

| Term | Meaning |
|---|---|
| **A-MEM / Mem0** | Production LLM-agent memory frameworks. They persist per-user distilled notes; that store is the artifact this paper protects. |
| **Anchors** | K random unit vectors defining the LSH buckets. Public-seed → free. |
| **AUC (ROC-AUC)** | Attack quality: **0.5 = chance (no leakage)**, 1.0 = perfect re-identification. |
| **Centroid** | The (normalised) average of the note-embeddings in one bucket. The released object. |
| **Clipping** | Rescaling a user's payload down to a maximum length `C`. What makes sensitivity **exact**. Applied **globally**, to the whole concatenated payload — see Remark 1. |
| **Composition** | Releasing repeatedly **adds** ε. The dominant term for a memory pool (§5.4). |
| **Cosine similarity** | Angle between two arrows. The only retrieval operation used. |
| **Decode-extract** | Closed-set reconstruction attack: pick the true in-bucket note from a supplied candidate set. Shown to be the **wrong instrument** (§7.3). |
| **δ (delta)** | The probability the DP guarantee simply fails. δ = 10⁻⁵ here; must be ≪ 1/N. |
| **DP (Differential Privacy)** | The output would be almost as likely had any one user been absent. |
| **ε (epsilon)** | The privacy budget. Smaller = more private. **ε ≈ 9.3** here, end-to-end. |
| **Embedding** | A sentence rendered as a vector of numbers, so that meaning becomes geometry. |
| **Exponential mechanism** | The DP primitive for privately *choosing* an option: sample with probability ∝ exp(ε·score/2). Used to pick the clip bound `C`. |
| **Evidence-recall@5** | Did the query route to the bucket holding its ground-truth evidence, within the top 5? |
| **Honest-but-curious** | An adversary who follows the protocol but analyses everything it sees. The server. |
| **K** | Number of buckets = memory **fidelity**. The paper's main dial. **K = 32 recommended.** |
| **LiRA** | Likelihood Ratio Attack. The strong, **calibrated** MIA: per-target, using shadow releases with and without that exact note. |
| **LSH** | Locality-Sensitive Hashing — a hash where *similar inputs collide on purpose*. |
| **MEXTRA** | Published prompt-injection attack that makes a live agent recite its raw-text memory verbatim. |
| **MIA** | Membership Inference Attack: "was this person's data in the pool?" |
| **M_min** | Survival floor. Noise shares are sized to it, so any turnout above it **over**-noises rather than silently under-noising. |
| **MRMMIA** | Published membership-inference attack against agent memory. |
| **Post-processing immunity** | Anything computed from a DP output is still DP. You cannot leak more by thinking harder. |
| **Projection (P)** | 384-d → d-d dimension reduction. **Public-seed Gaussian random**, deliberately *not* PCA (§7.2). |
| **Quantisation** | Real vectors → integers, so the crypto and the discrete noise work. Costs ~zero ε at range_max = 10⁶. |
| **RDP (Rényi DP)** | A reformulation of DP that composes cleanly and converts to (ε, δ). The accountant's working language. |
| **Retention@ε** | utility(private) / utility(clean). **Isolates the cost of the mechanism**, not of the architecture. |
| **Routing prior** | What the pool actually is: a map of *where the fleet's knowledge lives*, not a note exchange. |
| **SecAgg** | Secure Aggregation: the server learns Σxᵤ and **nothing else**, via pairwise zero-sum masks. |
| **Sensitivity (Δ₂)** | How much the answer can move when one user is added/removed. What the noise is calibrated to. |
| **Shadow release** | A re-run of the aggregation with a target note deliberately in or out — the calibration data LiRA needs. |
| **Skellam** | Difference of two Poissons. **Discrete** (survives crypto) and **closed under addition** (so distributed shares compose to exactly the central noise). The heart of the mechanism. |
| **σ (sigma)** | Noise scale. **σ = 0.606 → ε ≈ 9.3.** |
| **TPR@1%FPR** | Fraction of members an attacker confidently identifies while falsely accusing only 1% of non-members. **Floor = 0.01.** The metric the paper headlines. |
| **Vector-only release** | Release the sum-vectors but **not** the counts. Identical retrieval, **~1.5× tighter ε**, because count magnitude cancels under normalisation. |

---

## 10. Map from concept → code

| Concept | Where it lives |
|---|---|
| Skellam noise, SecAgg masks, quantisation, RDP accountant | `qpriviot_fl/privacy_utils.py` — `apply_distributed_skellam_noise`, `skellam_noise`, `generate_zero_sum_masks`, `quantize`, `skellam_rdp_epsilon`, `dp_quantile_clip` |
| Global clip (Remark 1) | `clip_rows_to_norm(V_u.reshape(1,-1), C)` in `scripts/agentmem/_agentmem_leakage.py` |
| Utility (real corpus) | `scripts/agentmem/_longmemeval_probe.py` |
| Utility (controls) | `scripts/agentmem/_agentmem_probe.py` (`make_projector`, `PUBLIC_PROJ_SEED`) |
| Utility (distilled) | `scripts/agentmem/_longmemeval_distilled_utility.py` |
| Leakage — weak cosine proxy | `_longmemeval_leakage.py`, `_agentmem_leakage.py` |
| Leakage — **calibrated LiRA** | `scripts/agentmem/_agentmem_lira.py` |
| Leakage — live agent (MEXTRA/MRMMIA) | `scripts/agentmem/_agentmem_llm_attack.py` (needs local `ollama serve` + `qwen2.5:7b`) |
| Joint frontier + multi-round | `scripts/shell/rerun_joint.sh`, `scripts/agentmem/_joint_frontier.py` (`sigma_for()`) |
| Accounting / discretisation-free check | `scripts/agentmem/_skellam_accounting.py` |
| Occupancy channel | `scripts/agentmem/_occupancy_channel.py` |
| Decode chance floor | `scripts/agentmem/_decode_floor.py` |
| Distillation (turns → notes) | `scripts/agentmem/_longmemeval_distill.py` |
| Figures / tables | `scripts/agentmem/rerun_figures.py`, `rerun_tables.py` |

**Key flags that reproduce the paper's own retractions:**

- `--proj {randproj,pca,publicpca}` — `randproj` is the honest default; **`--proj pca` reproduces the
  superseded, data-dependent grid bit-identically**, which is how the §7.2 ablation isolates the projection
  as the only variable.
- `--clip-eps` (default `0.1`) — the DP clip selection. **`--clip-eps 0` reproduces the superseded,
  unaccounted empirical p95.**
- `--gate {none,oracle}` (default `none`) — `none` releases **every** bucket, so an empty one carries
  normalised noise; this is the mechanism of Algorithm 1 and what **every** retrieval number is measured
  under. `oracle` suppresses empty buckets using the **clean counts** — a non-private read, i.e. an
  unaccounted release — and reproduces the superseded high-K cells.

**Full reproduction:**

```bash
bash scripts/shell/rerun_final.sh \
  && bash scripts/shell/rerun_joint.sh \
  && python scripts/agentmem/rerun_figures.py --grid rerun_grid_rp

bash scripts/shell/rerun_lira.sh     # the calibrated attack (§7.1)
# live-agent attacks need: ollama serve, with qwen2.5:7b pulled
```
