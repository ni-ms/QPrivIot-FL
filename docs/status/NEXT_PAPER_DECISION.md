# Next-Paper Decision Brief (2026-07-01)

Decision doc after the QFL "quantum helps privacy" direction died empirically and its
fallback ("d governs the DP crossover") was found **scooped**. Companion: `QFL_STATUS.md`
(full result tables), memory `project_qpriviot_fl.md`.

---

## 1. Situation in one paragraph

The distinctive QFL claim (a tiny-d VQC is *usable-private where the CNN collapses*) is
**dead**: across 10-class, 5-class, ε=8 and ε=3, the CNN wins absolute private accuracy at
every N and never collapses to random. The surviving claim ("DP cost scales with model
dimension d, setting an N-crossover") is **textbook prior art** — Bassily-Smith-Thakurta
(FOCS 2014, √d/(nε)) and **Chen et al. ICML 2022** (arXiv:2203.03761), which proves the
exact d-vs-N crossover *in this SecAgg + distributed-DP setting* as `min(n²ε², d)`. The
quantum element is decorative because the DP analysis is classical and model-agnostic.
**Reviewer kill line:** "re-demonstrates the established BST/Chen crossover on MNIST; no new
privacy phenomenon." → QFL paper is **workshop-tier at best**.

**Universal blocker across every option below:** everything is **single-seed (seed=42)** and
reported as **peak-over-rounds** accuracy. Both must be fixed for any archival venue.

---

## 2. The candidate papers (from the salvage scan)

| # | Paper | Speed | Venue | Novelty status |
|---|-------|-------|-------|----------------|
| P1 | **DP-FL evaluation pitfalls / reproducibility** (peak-metric inflation + phantom-privacy σ-deflation bug, in SecAgg+Skellam regime, + reproducer) | ~5–9 d | TMLR / ReScience C / TrustML workshop | Novelty NOT pre-empted (Finding B fresh); **best bet** |
| P2 | **d-governs-DP-cost characterization** (controlled d-sweep, VQC as extreme-low-d anchor) | ~6–10 d | IEEE QCE / TQE | Instantiation of known theory; honest salvage of QFL work |
| P3 | **Merged N-crossover + d-sweep** ("two axes of DP cost") | ~8–12 d | IEEE QCE / TQE / Access | Stronger single paper; still characterization |
| — | FEMNIST N-crossover **solo** | — | — | Too thin — phase transition is Geyer et al. 2017; fold into P3 |

**Hard cut:** do NOT write any "quantum helps privacy" paper — already disproven; a reviewer
finds it in one experiment.

---

## 3. Clarifications you asked about

### 3a. Venue realism & timelines (as of 2026-07-01)
- **TMLR** — rolling, no deadline; judges *correctness + usefulness over novelty* → the natural
  home for **P1**. Feeds MLRC/NeurIPS reproducibility (MLRC eligibility ~Sep 30 2026).
- **ReScience C** — rolling; literal scope fit for P1.
- **TrustML / reproducibility workshops** (NeurIPS/ICML/ICLR) — low-risk "plant the flag" first
  outlet for P1; CFPs land spring–summer 2026.
- **IEEE QCE 2026 (Quantum Week)** — technical-track deadlines (Apr 2026) have **PASSED**; only
  the **workshop track (~Jul 9 2026)** is catchable and it's **~8 days out** → too tight for a
  from-scratch write. Realistically QCE is a *next-year* target for P2/P3.
- **IEEE TQE** — rolling journal, **no deadline** → the practical home for P2/P3.
- **PoPETs / PETS** — next cycle deadline ~Aug 31 2026 (~2 months) — feasible for a *hardened*
  (multi-seed) P1/P3; single-seed will be rejected here.
- **IEEE SaTML 2027** — deadline ~Sep 2026 (~2 months) — good fit for P1 (security-of-ML).
- **FL@NeurIPS 2026 workshop** — CFP ~Aug–Sep 2026; fallback for P3.

**Takeaway:** the only *rolling, no-deadline* homes are **TMLR/ReScience (P1)** and **TQE
(P2/P3)** — so speed is bounded by *your* writing pace, not a CFP. The ~2-month targets
(PoPETs, SaTML) are realistic **only** after the multi-seed hardening.

### 3b. Effort vs. payoff (what the day counts actually contain)
- **P1 (5–9 d):** ~2 d harness cleanup into a drop-in reproducer · ~2 d multi-seed reruns
  (background CPU, cheap on small configs) to make the peak-vs-final inflation statistically
  solid · ~3–4 d writing. **Payoff:** an accepted paper at a venue that *rewards exactly what
  you have*; lowest novelty risk. **Risk:** Finding B must be framed as more than "selection
  bias is known" (see 3c).
- **P2 (6–10 d):** result + figures already done · ~2–3 d seeds on the d-sweep · ~4–5 d writing
  + careful positioning against BST/Chen. **Payoff:** salvages the weeks of QFL work into an
  honest characterization. **Risk:** reviewer wants more than "we instantiated a 2022 theorem."
- **P3 (8–12 d):** P2 + the FEMNIST panel; each panel needs seeds. **Payoff:** the strongest
  *single* mid-tier IEEE paper. **Risk:** still characterization, not discovery; more surface
  area to harden.

### 3c. Is P1's novelty *really* safe? (pressure-test)
- **Finding A (σ-deflation "phantom privacy" bug)** is a **known bug *class*** — Tramèr et al.
  "Debugging Differential Privacy" (arXiv:2202.12219); "Finding Private Bugs" (ICLR'23). If
  framed as a *discovery*, it gets torched. Use it as a **case study**, not a headline.
- **Finding B (peak-round accuracy inflation under DP)** is the **fresher lever** — the general
  fact (max-over-checkpoints is optimistically biased) is textbook, but the *DP-specific,
  quantified* claim that DP's added variance makes peak reporting **systematically inflate
  reported DP accuracy and distort privacy-utility conclusions** is not cleanly staked out.
- **To make B robust, not thin:** (1) tie it to a *privacy* consequence — peak reporting makes
  DP look *cheaper than it is*, biasing the tradeoff curves the whole field publishes;
  (2) show it **flips a qualitative conclusion** (we already saw the risk: CNN "40.5%" at N=10
  is a lucky spike on a curve sitting ~25% — under final-metric a different model can "win" the
  private cell); (3) ship it **paired with Finding A + the reproducer + the new SecAgg+Skellam
  regime**. **Verdict:** P1 is safe as *"two pitfalls + harness + new regime,"* riskier as
  *Finding B alone.*

### 3d. Is "publish fast" even the right goal?
Depends on your actual constraint — this changes the pick:
- **Thesis chapter / substantive contribution:** favor **P3** (merged, more substantial) or a
  solid **TMLR P1** — both read as real chapters.
- **Quick accepted paper for a milestone (funding/visa/progress report):** **P1 → workshop/TMLR**
  is the fastest defensible flag-plant.
- **CV line at a recognized venue:** P1 at SaTML 2027 or TMLR.
- **(Open question for you — tell me which of these you're optimizing for; the rest of the plan
  keys off it.)**

### 3e. The hybrid path (recommended if undecided)
The **multi-seed + final-metric hardening is a prerequisite for P1's core stat AND for any
archival venue for P2/P3** — so it is *never wasted*. Do it **first** as a shared step
(~1–2 background CPU-days), then commit with error bars in hand. This keeps P1/P2/P3 all open
at essentially zero opportunity cost.

**Concretely:** re-tool the scripts to log *final + mean±std* (not just peak) → 3-seed reruns
of the small configs (d-sweep, 5-class, FEMNIST) in the background → re-aggregate peak-vs-final
tables → revisit the choice.

---

## 4. NEW direction — federated concept into LLM / agentic space

**Core insight from the scan:** your Skellam-under-SecAgg is a **distributed-discrete-DP
primitive looking for a payload other than gradients**. The mechanism community
(Skellam/discrete-Gaussian + SecAgg) and the LLM-agent community are **still disjoint** — no
paper points distributed-discrete-DP at agent memory, retrieval statistics, preference vectors,
or skill/trajectory vectors. Novelty lives in the **payload + threat model + a dimension/
quantization argument**, never in the crypto — which plays *directly* to your "DP cost governed
by d" expertise, because agent payloads (memory embeddings, preference vectors, retrieval
histograms) are **tiny-d and often already discrete** — exactly where Skellam's low-precision
advantage compounds. Generic "federated DP-LoRA / prompt tuning" is **saturated — avoid.**

### Ranked directions (novelty × feasibility)

**★ THE BET — D1+D5: Private federated agent-memory aggregation with a leakage-drop endpoint.**
Aggregate per-user distilled agent memories / skill-embeddings across users under SecAgg +
Skellam-DP so the shared memory pool carries one central-DP guarantee and no individual trace
leaks; **evaluate by the measured drop in extraction / membership-inference attack success.**
- *Reuses:* the Skellam+SecAgg path **verbatim** (payload = memory-note embeddings), non-IID
  per-user partitioning, and the d-crossover analysis where it's *most* credible (memory
  embeddings are genuinely tiny-d).
- *New code (minimal):* wrap A-MEM / Mem0 memory-note embeddings as the real vector → existing
  quantize→Skellam→SecAgg→sum path → retrieval eval on the noised pool. **No model training.**
- *Novelty:* PARTIALLY-OPEN, leaning open. Neighbors are access-control (*Collaborative Memory*
  arXiv:2505.18279), on-device masking (MemPrivacy 2605.09530), or gradient-only Skellam
  (NeurIPS'21 2110.04995) — **none** fuse SecAgg+discrete-DP with agent-memory objects; the 2026
  survey (2604.16548) explicitly names "DP in federated agent adaptation" as a gap.
- *Experiment:* LoCoMo / LongMemEval split per-user; aggregate memories under SecAgg+Skellam;
  report retrieval/QA utility vs ε **and** MEXTRA (2502.13172) extraction / MRMMIA (2605.27825)
  membership advantage dropping. Attack-measurement side is already 3-deep — **cite/reuse, don't
  claim** (that's D5, folded in as the eval harness, not a separate paper).
- *Venue / effort:* NeurIPS/ICLR workshop, PPAI/SaTML, or IEEE mid-tier. **~4–6 weeks.**
- *Why bet on it:* maximizes the product — verbatim code reuse × signature d-analysis at its
  most credible × a crisp *positive, measurable* endpoint (attack-success ↓) × a demonstrably
  open spot.

**② SAFER FALLBACK — D2: Federated DP RAG over retrieval-statistic histograms.**
SecAgg-sum LSH/bucket-vote histograms with Skellam noise → shared retrieval datastore that is
central-DP with no trusted curator.
- *Reuses:* Skellam **closure-under-summation** on payloads that are *already integer counts* —
  the most literal Skellam fit of all five seams; multi-KB non-IID partitioning.
- *Novelty:* PARTIALLY-OPEN. Closest competitor **DP Datastore Generation** (arXiv:2606.01413,
  2026) is **centralized, single datastore, no SecAgg, plain additive noise** → your federated +
  SecAgg + composes-under-summation framing is a clean delta. **Risk:** a "federated follow-up"
  to 2606.01413 is the likely scoop — move fast.
- *Experiment:* KILT / HotpotQA / FRAMES sharded into distributed KBs (multi-hop = naturally
  multi-source); recall + QA vs ε. Cleanest, least-LLM-tooling Skellam demo. **~4–6 weeks.**
  ACL/EMNLP Findings or security workshop.

**③ D4 (gap exists, but hot & heavy): SecAgg+Skellam over agent trajectories/skill vectors.**
Real opening — *Fed-SE* (arXiv:2512.08870) lists DP as future work — but fastest-closing space
and trajectory payloads are higher-d / messier to quantize than histograms. More engineering.

**Crowded — do NOT frame around these:** federated DP-LoRA / prompt tuning (DP-FPL 2501.13904,
FedDTPT, DP-OPT — saturated); preference-vector aggregation is near-scooped by **POPri**
(arXiv:2504.16438: SecAgg + user-level DP at ε=1/7 on preference vectors — but *continuous
Gaussian, no Skellam*, verified); agent-memory-leakage *measurement* is 3-deep (MEXTRA, MRMMIA,
AgentLeak — cite, don't claim).

**Timing caveat:** all five sit in fast-moving 2026 arXiv space; competitors are circling
(2606.01413, the 2604.16548 survey). Target a workshop / IEEE-mid-tier deadline in **6–8 weeks**.

---

## 5. Recommendation

Two genuinely good moves, on different time/novelty horizons:

- **Fast, safe, this month:** **P1** (DP-FL evaluation pitfalls), reframed per §3c → TMLR /
  ReScience / TrustML workshop. Lowest novelty risk, reuses data you already have.
- **Higher ceiling, fresh line, ~6 wks:** **D1+D5** (federated agent-memory aggregation with a
  measured leakage-drop endpoint) → workshop / IEEE mid-tier. Reuses your Skellam+SecAgg code
  *verbatim* on a novel payload and puts your d-crossover analysis where it's most credible; the
  attack-success-↓ endpoint makes it a *positive, measurable* result in an open spot.
- **No-regret first step regardless:** the **multi-seed + final-metric hardening** (§3e) — it's
  a prerequisite for P1's core statistic *and* every archival venue, so it's never wasted. Kick
  it off in the background before committing.

**The tension to resolve (your call):** P1 is the safe flag-plant; **D1+D5 is the better *bet*** —
it's the only option that turns the whole federated-privacy toolkit into a *forward-looking*
agentic-LLM line rather than closing out the DP-FL work. If you want momentum into a hotter
area and can spend ~6 weeks, D1+D5. If you need an accepted paper fastest, P1. They're also
**sequenceable:** P1 now (cleans the harness, forces the multi-seed hygiene), D1+D5 next
(inherits the cleaned harness + reproducer).