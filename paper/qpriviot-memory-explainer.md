# QPrivIoT-FL — Plain-English Walkthrough

*A section-by-section explainer for "Private Federated Aggregation of LLM-Agent Memory: Usable Shared Memory with a Measurable Leakage-Drop Guarantee under Secure Aggregation and the Skellam Mechanism."*

---

> **⚠️ Updated.** This explainer tracks the **revised** paper (joint-frontier version). Two things changed materially from the earlier draft: (1) leakage and utility are now measured **at the same operating point** — the old draft measured them at non-overlapping settings; (2) the paper now accounts for **repeated release**, since a memory pool is re-aggregated over time. Both changes made the results *better*, not worse. See "What changed and why" at the end.

## The one-sentence version

**AI agents remember things about you; pooling those memories across users would make every agent smarter, but memories are exactly the sensitive stuff — this paper shows you can pool them anyway, with a mathematical privacy guarantee, and it barely hurts how useful the pool is.**

---

## Title & Abstract

The title glues four ideas together: *shared agent memory*, *made private*, *with a measurable drop in leakage*, using *two specific tools* (Secure Aggregation and the Skellam mechanism).

The abstract's three claims are the whole paper:

1. The private shared memory still works — it keeps **95% of the retrieval quality** of a non-private pool.
2. Attacks that used to identify people **stop working** — an attacker's success drops from near-certain (AUC 0.93) to a coin flip (0.53).
3. A **warning to other researchers**: a common preprocessing step (fitting a dimension-reduction on your own users' data) secretly leaks privacy *and* fabricates false findings. The author's own earlier draft fell for this, and retracts it here.

That third one is unusual — the paper openly kills one of its own earlier claims.

---

## §1 Introduction — why anyone should care

**Agent memory is the new sensitive payload.** Tools like A-MEM and Mem0 give each user's AI agent a little notebook: *"allergic to penicillin," "prefers metric units," "is migrating a Postgres cluster."* That notebook is what makes the agent feel personal. It is also a dense pile of personal information.

The tempting idea: **share the notebooks across users**, so if one person's agent figured out a tricky fix, everyone's agent benefits. The problem: you can't just dump everyone's notes into a shared pile. Two published attacks already break that:

- **MEXTRA** tricks an agent into reciting its memory word-for-word.
- **MRMMIA** figures out *whether a specific person's data is in there* (membership inference — a privacy violation on its own; think *"was this person in the HIV-clinic dataset?"*).

**Why existing defenses don't cut it.** Current approaches either control *who is allowed to read* a note (access control) or *blank out names* in notes (redaction). Neither gives a guarantee about the pool *as a whole*. And the one approach that does add real math puts a trusted central authority back in charge — the exact thing we were trying to remove.

**The approach in one breath:** every user turns their notes into number-vectors, sorts them into a shared set of bins, adds up the vectors in each bin, sprinkles in calibrated random noise, and sends the result through a cryptographic protocol where the server can compute the *total* across all users but can never see any single person's contribution. What comes out is one shared "average memory" per bin, with a formal privacy guarantee attached.

**The honest framing (worth repeating):** the *crypto is not new*. It is lifted verbatim from prior federated-learning work. What's new is (a) pointing it at a new payload — agent memory instead of model gradients, (b) actually measuring that real published attacks stop working, and (c) the negative result.

---

## §2 Related Work — who did what already

Three threads:

- **Federated DP** (federated learning + Secure Aggregation + discrete noise) exists, but has always been used to train *models*. This paper reuses that machinery for *retrieval* instead.
- **Agent-memory attacks** exist and are getting good. The paper doesn't invent attacks — it *borrows the attackers' code as its test suite*, which is the right move.
- **The nearest prior work** ("DP Datastore Generation") is structurally similar — hash into buckets, add noise — but it is *centralized* (one trusted curator sees everything) and uses ordinary continuous noise. This paper's version is curator-free, and that is what forces the choice of Skellam noise (below).

---

## §3 Threat Model — who's attacking, and what counts as winning

**The cast:** N users with agents; a server that aggregates; an attacker who is *whoever receives the shared pool* — the server itself, or any agent that queries it.

The server is **honest-but-curious**: it follows the protocol correctly, but it will absolutely read anything it can see. So the design makes sure it can't see anything useful.

**The two attacks they measure:**

- **Membership inference** — "Was *this person's* note in the pool?" Scored as ROC-AUC, where **0.5 = pure guessing, 1.0 = perfect**. This is the headline metric: get it to 0.5 and the attacker learned nothing.
- **Extraction** — "Can I reconstruct someone's actual note from the published average?"

**The key insight of the whole threat model — the "vulnerable tail."** When you average things together, popular bins are naturally safe: if 200 people contributed to a bin, the average tells you nothing about any one of them. The danger is bins where **only one or two people contributed** — there, the "average" basically *is* that one person's note. So the paper deliberately measures leakage on the worst case: bins with **≤ 3 contributors**. That is an honest choice; a less careful paper would report the flattering average across all bins.

**What they explicitly do *not* defend against** (stated up front — a credibility move):

- Malicious users who deliberately poison the pool to attack someone.
- The fact that *participating at all* is visible. (They specify a fix — everyone must submit, idle users send padded dummy data — but they don't test it.)

---

## §4 Bucketed Centroid Memory — the data structure

This is the core object:

1. Each note becomes a 384-number vector via a standard sentence encoder (similar meaning → similar vector).
2. Squash it down to a smaller size `d` (e.g. 32) using a **random projection**. **Critical detail:** the randomness comes from a *published seed*, so it is fixed, public, and — the point — it **never looked at anyone's data**. Anyone can regenerate it. It leaks nothing.
3. Sort each note into one of `K` **buckets** using random hyperplanes (LSH — locality-sensitive hashing; similar notes tend to land in the same bucket).
4. Each user adds up all their vectors within each bucket → one **sum-vector per bucket** per user.
5. The shared memory = for each bucket, sum everyone's sum-vectors, divide by the total count, normalize. That is the **centroid** — a bucket's "average idea."
6. To search: embed your query, compare it to the K centroids by cosine similarity, take the top few buckets.

Two knobs: **K** = how many bins (more bins = finer-grained memory), **d** = how many numbers per vector (resolution). Both trade off against privacy.

The paper flags something here that becomes a major theme: the *obvious* choice for step 2 is PCA, which finds the best directions **by looking at your data**. That is a trap. Hold that thought.

---

## §5 Private Release — the actual mechanism

### 5.1 Clipping and quantisation

Before adding noise you must bound how much *any one user* can influence the result. So each user's **entire payload** (all K buckets stacked into one long vector) is **clipped** — if its length exceeds `C`, scale it down to `C`. Then it is rounded to whole numbers, because the crypto only works on integers.

Why this matters: clipping to `C` means removing one user changes the total by *at most* `C`. That number — the **sensitivity** — is exactly what determines how much noise you need. No clip, no guarantee.

### Remark: "why the clip must be global" — the near-miss bug

One of the best bits to mention, because it's a real trap they caught.

If you clipped each *bucket* separately to `C`, a user with notes in 21 different buckets could contribute `C` to *each one* — so their true influence is √21 × C, not C. An accountant still assuming `C` would be **under-counting the privacy cost by a factor of ~5**.

They quantify it: the reported guarantee of **ε ≈ 9.3 would actually have been ε ≈ 69** (and **159** at high K). That is not a rounding error — that is a broken guarantee, silently. Their code clips globally, so they're fine, but they document it as a warning.

### 5.2 The clip bound must be public too

Subtle and clever. The clip bound `C` gets shipped to every client, so **it is part of the public output**. But the natural way to pick it — "use the 95th percentile of my users' actual data sizes" — means `C` is *computed from private data*. Publishing it is an unaccounted leak. (Their earlier draft did exactly this.)

Fix: pick `C` using the **exponential mechanism** — a standard DP tool for making a choice privately. It costs a tiny bit of budget: **ε_c = 0.1, which is 0.2% of the total**.

They're honest about the price: the private picker is *conservative* — it overshoots (picks 21.7 when the true answer is 13.7), which means **too much noise**, which costs accuracy (recall 0.416 → 0.406). They call that ~2% cost *"the correct price for an accounted guarantee."*

A freebie alongside it: the quantiser bound `B` can just be set to `C`; they prove it never clips anything and costs zero privacy.

### Skellam noise + SecAgg — the two-part trick

**Secure Aggregation (SecAgg):** every user adds a secret mask to their data. The masks are constructed so that **when you add all users' data together, all the masks cancel to zero**. The server therefore recovers the exact *total* while seeing each individual submission as gibberish.

> *Analogy:* ten people want to know their average salary without revealing individual salaries. Each adds a huge random number to their salary; the random numbers are pre-arranged to sum to zero. Add up the ten scrambled figures — the randomness vanishes and the true total appears. No one ever saw a salary.

**The Skellam mechanism:** each user adds a small amount of *discrete* random noise (Skellam = the difference of two Poisson draws — it produces whole numbers, which is what the integer crypto requires).

**Why Skellam specifically — the key technical reason:** *the sum of Skellams is itself a Skellam.* So if 500 users each add 1/500th of the required noise, the **total** carries exactly the right amount of noise — no more, no less. Nobody had to be trusted to add it centrally. **The distributed version literally equals the ideal centralized version.** Ordinary Gaussian noise would not survive the integer/modular arithmetic SecAgg uses. That is the entire reason this mechanism was chosen.

### 5.3 Vector-only release — a free 1.5× win

The natural design publishes two things: the summed vectors *and* the counts. Two published things = twice the privacy cost.

But: since you normalize the centroid at the end (scale it to unit length), **dividing by the count does nothing to the direction**. The count cancels out. So don't publish it. Same retrieval results "to the last decimal," and **ε drops from ~14.0 to ~9.3 — about 1.5× tighter, for free.**

**The catch they're honest about:** count *magnitude* cancels, but **occupancy** does not. If a bucket has *zero* contributors, its "sum" is pure noise — and normalizing pure noise produces a perfectly respectable-looking unit vector that competes for top-k slots against real centroids. Cosine similarity cannot tell them apart. Handled in §7.9.

### 5.4 Privacy accounting

They compute ε with the exact published formula for Skellam noise, converted to the standard (ε, δ) form, with δ = 10⁻⁵.

Two things they defend carefully:

- **δ is compared against the number of *users* (500), not notes.** The rule of thumb is δ ≪ 1/N, and 10⁻⁵ is 200× safer than needed. Scaling to a million users would require tightening δ — they show it costs **less than 23% more ε**, so it scales fine.
- The mapping: noise scale σ = 0.606 → **ε ≈ 9.3**; σ = 1.615 → **ε ≈ 3.2**. Bigger σ = more noise = smaller ε = more private.

**What ε means, for your listener:** ε is a privacy budget; lower is more private. ε ≈ 9.3 is *not* the tight academic gold standard (purists want ε ≤ 1). The paper's defense is empirical — *at this ε, all the actual attacks are dead*, which is the thing you actually care about.

### 5.5 Algorithm 1

The recipe written out formally: server picks `C` publicly → each client embeds, buckets, sums, clips globally, quantises, adds its noise share, masks, submits → server adds everything up, masks cancel, normalizes → publishes K centroids → agents query by cosine. Nothing new; just assembled.

### 5.6 Client dropout — a landmine they flag

In real federated systems, phones drop off mid-protocol. If each of 500 clients added 1/500th of the noise but only 250 survive, **you only get half the noise you paid for** — and your "ε = 9.3" guarantee is silently actually **ε = 13.9**. Worse, it fails hardest exactly when participation is worst.

Two fixes offered: size each noise share to a *minimum survivor count* and abort if turnout falls below it (so the guarantee can only get *better*, never worse); or secret-share the noise seeds so dropped clients' noise can be reconstructed.

**They specify the fix but their experiments assume 100% participation — and they say so plainly and list it as a limitation.** Don't claim otherwise.

---

## §6 Experimental Setup

- **LongMemEval (oracle)** — real long-conversation memory. 500 users, ~11k notes, 479 queries, with ground truth for which turns held the evidence.
- **LongMemEval (s)** — same 500 users but **246,073 notes** (~96% irrelevant distractors); the "does this hold at scale?" check.
- **LLM-distilled notes** — they ran a local LLM (qwen2.5:7b) over the dialogues to produce actual A-MEM/Mem0-style memory notes. **This is the realistic payload** — what a real agent would store.
- **Controls** — 20-Newsgroups, for clean sweeps.
- **Protocol** — 5 random seeds, report mean ± standard deviation, and use the **final** metric, not the best-looking one. (Reporting "peak" is a classic way to flatter yourself.)

---

## §7 Results

### 7.1 The joint frontier — the centerpiece

**This is the table to show someone.** Both axes — how useful the pool is, and how badly it leaks — measured at *the same* setting, for every K:

| K | retention @ε≈9.3 | LiRA AUC | **TPR@1%FPR** | decode-extract |
|---|---|---|---|---|
| **32** | **95%** | 0.833 → 0.502 | **0.175 → 0.011** | 0.771 → 0.396 |
| 64 | 86% | 0.908 → 0.505 | 0.357 → 0.011 | 0.667 → 0.188 |
| **128** | **72%** | 0.954 → 0.502 | 0.552 → 0.011 | **0.794 → 0.070** |
| 256 | 60% | 0.981 → 0.503 | 0.750 → 0.011 | 0.789 → 0.029 |
| 512 | 41% | 0.993 → 0.505 | 0.874 → 0.013 | 0.874 → 0.007 |
| 1024 | 25% | 0.997 → 0.504 | 0.945 → 0.013 | 0.912 → 0.003 |
| 2048 | 15% | 0.998 → 0.505 | 0.957 → 0.012 | 0.928 → 0.001 |

Three things to say about it:

1. **The privacy axis is flat.** DP drives the attack to the 1% false-positive floor at *every* K. Privacy doesn't trade against K at all.
2. **So you pick K purely on utility — and coarse wins.** Retention falls 95% → 15% as K grows. **K = 32 is the recommendation.**
3. **Membership and reconstruction don't die at the same K.** At K=32 membership is dead (TPR 0.175 → 0.011) but reconstruction only halves (0.771 → 0.396). If reconstruction is your threat, run **K = 128**, where both hit the floor, at 72% retention. The paper reports both rather than quoting the flattering one.

**Density is everything** — coarse buckets pool more users, so DP is nearly free; fine buckets starve, so noise bites.

### 7.2 The negative result — "the dimension crossover is an artifact"

The most intellectually interesting section.

The earlier draft claimed something exciting: making `d` small *simultaneously* reduced leakage **and** improved DP utility — a free lunch, tiny-`d` as the sweet spot on both axes.

It was **an artifact of using PCA fitted on the users' own notes.**

The mechanism is genuinely elegant. PCA finds the directions where *your specific corpus* varies most. Keep only a few of them and you (a) keep exactly what everyone has in common — which flatters retrieval scores — and (b) **throw away the idiosyncratic quirks that make an individual identifiable** — which flatters privacy scores. Both axes improve. It looks like a discovery. But it is paid for with **information about the private corpus, spent outside the privacy accounting** — it was *secretly anonymising the data for free*, and free is exactly what privacy never is.

They re-ran everything three ways — leaky PCA, honest random projection, and PCA fit on a *public* corpus — holding everything else fixed. **And they built in a control:** at d = 384 all three projections are mathematically the identity function, so all three *must* produce identical numbers. They do. That proves the harness is sound and the projection is the only variable.

Result: **both halves of the "discovery" evaporate.** Under an honest projection, tiny-`d` goes from the *best* choice to the *worst* one on utility, and the leakage gradient flattens from 0.944 → 0.989 down to 0.988 → 0.989 (i.e. nothing). And after DP, leakage sits at 0.50–0.55 **for every dimension and every projection** — so dimension was never the operative variable anyway.

**The broader warning (the "it travels" claim):** data-dependent preprocessing — PCA, whitening, vocabulary selection, learned quantisers, clip bounds read off the data — is *standard practice*, is *an unaccounted privacy leak*, and can *manufacture findings out of nothing*. The identity-projection control that catches it **costs one extra run**.

### 7.3 The weak attack is blind exactly where you'd deploy — the second negative result

This is new, and it's the best story in the paper after the projection one.

The **cosine-similarity attack** is what the agent-memory literature reaches for by default. On the *identical clean release* at K = 32:

- **Cosine proxy says: AUC 0.504.** Chance. Nothing to see. The pool is safe.
- **Calibrated LiRA says: AUC 0.833, 17× the false-positive floor, 77% of notes reconstructed.**

The pool was **never safe.** The proxy just couldn't see it — because an uncalibrated cosine score compares you against a *population* baseline, and at coarse K everyone looks close to a centroid that averages ~340 notes. Only a *per-target* likelihood ratio can separate "close because the bucket is broad" from "close because I'm in it."

Worse, the proxy **manufactures a false story about where leakage lives.** Its sensitivity grows with K, so a proxy-only evaluation concludes leakage is a *sparse, high-K, corner-case* phenomenon — present only where retention is 15–41% and nobody would ship anyway. **That is exactly what the earlier draft concluded, and it was wrong twice**: it understated the threat where you'd actually deploy, and it invited readers to dismiss the mechanism as solving a non-problem.

**The shape this shares with the projection result:** a *default choice* — the obvious projection (PCA), the obvious attack (cosine) — that quietly biases the evaluation toward *"the mechanism is fine."* Both were caught by one cheap control run.

### 7.4 The realistic payload makes it *better*

LLM-distilled notes (the real thing an agent stores) leak **more** than raw dialogue, because distillation *concentrates* the identifying facts — clean tail-AUC rises 0.93 → 0.96. **And DP still kills it.** Utility at 500 users holds at **89% retention**.

Sub-finding: the 100-user pilot retains only 65%, but 500 users retains 89% — **not because of scale, but because of density** (more contributors per bucket). Same lesson as §7.1.

### 7.5 Accounting — where the ε actually goes

The most quotable table in the paper. Total ε ≈ 9.31, decomposed:

| Term | ε | Note |
|---|---|---|
| Base noise (Gaussian-RDP reference) | 9.2909 | — |
| + making it discrete (for the crypto) | 9.2909 | **+0.0000000013 — free** |
| + the projection | 9.2909 | **+0.000 — free**, it's public-seed |
| + privately picking the clip bound | **9.3109** | **+0.020 — 0.2%** |

The point: **the reported ε covers the entire pipeline, not just the last step.** Most papers only account for the final noise addition and quietly ignore everything before it.

### 7.6 Robustness

The refactor that introduced the `--proj` flag was validated by re-running the old setting and confirming it reproduced the previous results **bit-for-bit to 10⁻⁹**. That is how they can claim the projection was the *only* thing that changed. Re-running everything at 5 seeds flipped no conclusions — **except the dimension crossover, which is the point of §7.2.**

### 7.7 At scale — 246,000 notes

On the 22×-bigger corpus, **DP costs nothing measurable** (97–100% retention), and **the vulnerable tail is completely empty** — every bucket has plenty of contributors, so membership inference is *already* at chance (0.50) **without any DP at all**.

They read this correctly, and it is a maturity signal: this **confirms and bounds** the headline rather than extending it. The dramatic 0.93 → 0.53 drop is a *sparse-data* phenomenon. In the dense regime, averaging already protects you and DP is free insurance. A less careful author would have spun this as "it scales!"; they instead say "it scales, and here's the honest reason the headline number is smaller here."

### 7.8 The calibrated attack (LiRA) — the strongest evidence

The cosine-similarity attack used so far is *weak*. So they run the modern gold standard: **LiRA**, a likelihood-ratio attack that builds dozens of "shadow" versions of the pool with and without each target note, and asks which world the observed pool looks like.

The result is dramatic in **both** directions:

| Metric | Clean pool | At ε ≈ 9.3 |
|---|---|---|
| AUC | 0.997 | 0.504 |
| **TPR at 1% FPR** | **0.945** | **0.013** |
| Note reconstruction | 0.912 | 0.003 |

Read the middle row out loud. "TPR at 1% false-positive rate" is the metric that actually matters operationally: *if the attacker is only allowed to be wrong 1% of the time, how many people can they correctly finger?* On the untreated pool: **94.5% of them.** On the private pool: **1.3% — which is literally just the 1% false-positive floor. The attack has no signal at all.**

It also **retroactively vindicates §7.3**: the calibrated attack lands at 0.504 where the weak cosine attack read 0.53–0.55. So that stubborn residual was never a real membership signal — it was noise in a bad attack.

### 7.9 The published attacks vs a live agent

They build an actual working agent (local qwen2.5:7b) with shared memory and run the two real published attacks against it end-to-end, swapping only the memory backend:

| Backend | MEXTRA (verbatim recovery) | MRMMIA (membership AUC) |
|---|---|---|
| Raw text (status quo) | **1.000** — every note recovered | **0.981** |
| Our private pool | 0.017 | 0.526 (coin flip) |

**Two pieces of intellectual honesty here — they are what make the paper trustworthy:**

1. **The 0.017 isn't a real leak.** The private pool contains *no text at all*, so verbatim extraction is *structurally impossible*. That 1.7% is the **false-positive rate of their own detector** — a public note that happened to look similar enough to a target. In their words: *"It measures our measurement, not the release."* They report it instead of rounding it to zero.

2. **They refuse to take credit for beating MEXTRA.** MEXTRA attacks *raw text*. Their pool stores no text. So the honest claim is not "we defeat MEXTRA" but "we **structurally immunise** against it — the thing the attack eats no longer exists." They explicitly call the alternative framing a **strawman**, and note that *any* vector-only store gets the same immunity for free. The attack a vector store genuinely must answer for is **embedding inversion** — reconstructing a note from the centroid — and that is §7.8's decode attack, run in its *strongest possible* form (hand the attacker the true candidate list; they only have to pick). 91% on clean, **0.3% on theirs**.

**They also found and reported a bug in their own attack harness.** The membership probe was drawing non-members from the same pool the agent was reading from, so non-members appeared verbatim in the agent's context while members never could — the attack scored *below* chance (0.31–0.35). They point out the tell: **any AUC reliably below 0.5 is a bug, not a privacy win** (an attacker would just invert their answer and get 0.65–0.69). They fixed the split and re-ran. The superseded draft carried the same confound.

### 7.10 Occupancy — the loose end, priced honestly

Back to the empty-bucket problem from §5.3:

- **At the settings they actually report (K ≤ 256), no bucket is ever empty.** The failure mode is *absent*, not tuned away. (At K = 2048, though, 23% of buckets are empty.)
- Detecting empty buckets *privately* is **expensive** — a second published channel costs roughly ε 9.3 → 13.9.
- Doing it for *free* (thresholding the vector's own length against the expected noise level) is **exactly at chance — it doesn't work at all.**

The reason is beautiful: **"occupancy and membership are the same signal."** The very noise that stops you telling whether *this specific person* is in a bucket is the same noise that stops you telling whether *anyone at all* is in that bucket. No mechanism can hide the member while revealing the bucket. It is not an engineering failure; it is information-theoretically the same question.

Their conclusion: don't buy a count channel — **use fewer, denser buckets**, which is what you wanted anyway for utility.

---

## §8 Discussion & Limitations

Essentially a list of "here's where we could be attacked, said before you say it." The four gaps they name:

1. **Client dropout** — specified, not measured.
2. **Occupancy at high K** — priced, not solved.
3. **δ at production scale** — costs < 23% more ε to fix.
4. **The small AUC residual** — real, but unusable (1.3% TPR at a 1% error budget).

And the line that sums up the paper's character:

> **"The cost of honesty, and what it buys."** Closing the two unaccounted leaks (the data-fit projection and the data-read clip bound) **lowered every clean utility number by about a quarter** (evidence-recall 0.577 → 0.428). They took the hit. In exchange, the ε now covers the whole pipeline — and, the payoff, **both privacy results got *stronger*, because the PCA they threw away had been pre-anonymising the corpus for free.** They kept the old (wrong) numbers in the paper as the ablation rather than quietly deleting them.

---

## §5.7 Repeated release — because a memory isn't released once

A model is trained once. A **memory keeps growing** — so a real deployment re-aggregates on some cadence, and the guarantee that matters is the *composed* one.

Naively re-releasing is ruinous: privacy budgets add up, so **daily re-aggregation for a month costs ε ≈ 93, not 9.3.** A mechanism advertised at 9.3 and re-run nightly is not a 9.3 mechanism.

But the fix is cheap, because the noise you need only grows as **√T**. So you *buy* the cadence up front:

| Re-aggregation schedule | noise σ | retention at the **same** ε ≈ 9.3 |
|---|---|---|
| once (the old paper) | 0.606 | 95% |
| quarterly for a year | 1.211 | 89% |
| **monthly for a year** | 2.098 | **81%** |
| daily for a month | 3.317 | 73% |
| weekly for a year | 4.367 | 66% |

**The headline: a pool that re-aggregates monthly for a full year costs 81% retention at the same ε you were already claiming.** That turns a one-shot demo into a deployable system.

---

## The 60-second version to say out loud

> AI agents build up a private memory about each user. Pooling those memories across users would make everyone's agent smarter — but memory is exactly the sensitive part, and published attacks can already pull verbatim facts out of it, or tell whether you're in the pool.
>
> This paper pools them anyway. Everyone turns their notes into vectors, sorts them into 32 shared bins, and sends in bin-totals through a crypto protocol where the server sees the *sum across all users* but never any individual's data. Each user also adds a pinch of a special kind of noise (Skellam) designed so all the pinches add up to exactly the right amount of blur — meaning nobody has to be trusted to add it.
>
> What you get back isn't other people's notes — it's a **shared map of what the fleet knows**, which your agent uses to route a question to the right place. No text ever leaves anyone's device.
>
> Result, measured **at one single setting**: the pool still finds the right thing **95% as often** as an unprotected one, while a state-of-the-art attack that could identify members at 17× the error floor — and reconstruct 77% of the notes — is driven to **exactly the floor. Zero signal.** Against a real live agent, verbatim extraction goes from recovering *literally every* targeted note to recovering none, because the pool contains no text to extract. And it survives a realistic deployment: re-aggregating monthly for a whole year still fits the same privacy budget, at 81%.
>
> And the twist — actually, *two* twists, and both are the author catching himself. First: an earlier version found that squeezing vectors to a tiny dimension improved privacy *and* utility at once. It was **fake** — an artifact of fitting the dimension-reduction on the users' own data, which secretly anonymised it for free. Second: the standard, obvious way everyone measures this kind of leak **reports "perfectly safe"** on a pool that a proper attack tears open. Both mistakes are the *default* thing to do. Both were caught by one extra control run.

---

## Questions to have ready

**"ε = 9.3 isn't a great privacy number, is it?"**
Correct, and the paper says so outright. Purists want ε ≤ 1. Two defenses. First, it's an **end-to-end** budget — it covers the projection, the anchors, the clip bound, everything. Most papers' smaller ε covers only the final noise step and silently excludes a data-dependent preprocessing stage of exactly the kind §7.2 proves is both leaky *and* distorting. So the numbers aren't comparable in the direction you'd assume. Second, the budget is a means: *at this ε, every attack they can mount is at its floor.*

**"Isn't the crypto just borrowed?"**
Yes, and they say so in three separate places. The contribution is the payload, the end-to-end accounting (including repeated release), the joint frontier, and the two negative results.

**"Wait — if the pool has no text, what does a receiving agent actually get?"**
The honest answer, and the paper now leads with it: a **routing prior**, not a note exchange. The pool is K direction-vectors that tell an agent *which region of memory space holds knowledge*, so it can route a query — to the right local memory, the right tool, the right public corpus. It does **not** hand you another user's fix. The paper explicitly declines to claim that, on the grounds that any mechanism that *did* deliver it would be releasing exactly the artifact the attacks eat.

**"What's the single biggest practical takeaway?"**
**Density.** Everything good comes from many contributors per bucket. Coarse buckets → DP is nearly free and the empty-bucket problem never arises. Fine buckets → buckets starve, noise dominates, every problem shows up at once.

---

## What changed and why (vs. the earlier draft)

Useful if anyone saw the previous version.

| | Old draft | Revised |
|---|---|---|
| **Utility & leakage** | Measured at **non-overlapping** settings (utility at K≤256, leakage at K≥512) and the two headline numbers quoted side by side as if from one release | **Joint frontier**: both axes at every K. The recommendation (K=32) is now defensible at a single operating point |
| **The attack** | Headlined an uncalibrated cosine proxy | Headlines the **calibrated LiRA**. The proxy is demoted — and its blindness is reported as a *finding* |
| **Leakage claim** | "A sparse-tail phenomenon; dense buckets are protected by averaging anyway" | **Withdrawn.** The clean pool leaks at *every* K; the proxy just couldn't see it. DP kills the attack at every K too |
| **Repeated release** | Not addressed at all — not even in the limitations | New §5.7: naive re-release costs ε≈93/month; a bought cadence gives **monthly-for-a-year at 81%** |
| **What the pool *is*** | Implied cross-user knowledge transfer ("every agent inherits the fix") | Explicitly a **routing prior**, no text, narrower claim, actually delivered |

**The pattern worth noticing:** every single one of these changes made the paper's *results* stronger, not weaker. Being honest about the two disjoint grids revealed a better joint result. Being honest about the weak attack revealed a real leak that DP defeats. That's not a coincidence — a defence that only survives weak adversaries usually fails on the first strong one, and this one did the opposite.

---

## Glossary

| Term | Plain meaning |
|---|---|
| **Embedding** | A list of numbers representing a piece of text; similar meanings → similar numbers. |
| **Bucket / LSH** | A bin. Similar notes are hashed into the same bin. |
| **Centroid** | The average vector of everything in a bucket — the "shared memory" that gets published. |
| **SecAgg** | Crypto that lets a server compute the *sum* of everyone's data without seeing anyone's individual data. |
| **Skellam noise** | Whole-number random noise. Special property: adding lots of small Skellams gives you one exact big Skellam — so noise can be added distributively, with no trusted party. |
| **Differential privacy (DP)** | A math guarantee that the output barely changes whether or not any one person participated. |
| **ε (epsilon)** | The privacy budget. Lower = more private. |
| **Sensitivity** | The most one person can shift the result. Determines how much noise you need. |
| **Clipping** | Capping each user's contribution so sensitivity is bounded. |
| **Membership inference (MIA)** | Attack: "was this person in the data?" |
| **AUC** | Attack score. 0.5 = coin flip (no leakage), 1.0 = perfect. |
| **TPR@1%FPR** | The realistic attack score: how many people can the attacker correctly identify while being wrong only 1% of the time? |
| **LiRA** | The state-of-the-art membership attack; much stronger than a naive similarity check. |
| **MEXTRA / MRMMIA** | The two published attacks on agent memory: verbatim extraction, and membership inference. |
| **Retention** | Private utility ÷ clean utility. 95% = the private pool is nearly as good. |
