# QPrivIoT-FL — Paper Status & Findings

_Last updated: 2026-06-17 (FEMNIST confound-resolution grid — FINDING 3)_

A consolidated status of the QPrivIoT-FL paper: what was tried, what died, what the
literature says, and the current plan. Intended as a working hand-off document.

> ## Status at a glance (2026-06-17)
> **Goal:** a *characterization + deployable-system* paper for a solid IEEE venue (IoT-J /
> Access / networking conf), **not** a new-algorithm paper — every novel-mechanism direction
> (adaptive per-layer DP, per-device fairness, DP-Ditto personalization, DP-LoRA) was ruled out
> over 4 literature scans as saturated/out-of-budget.
> **Headline (now confirmed on 2 datasets):** the **client-count crossover** — *distributed*
> Skellam DP **under SecAgg** is unusable cross-silo (N≈10) but climbs to usable accuracy as
> clients scale; every **naive per-client** baseline (float DP without SecAgg, *and* integer
> DP under SecAgg) stays at ~random for **every** N. On MNIST this is a *window* (upper collapse
> at N=200 from data starvation, a fixed-dataset confound); on **FEMNIST** (writer-partitioned,
> constant per-client data) it is a clean **monotone** crossover — DP cost shrinks 76→7.6pp as
> N→200 with no upper collapse, which **resolves the confound** and is the paper's central
> figure.
> **Phase:** past the go/no-go on the *characterization* paper — but actively evaluating a
> higher-novelty pivot (the team finds pure characterization too thin). See §8 (novelty scan
> 2026-06-18): the leading candidate is **DP + Quantum FL**, where the crossover shifts left
> because VQCs have tiny `d` — a logically-grounded, harness-reusing extension of the crossover
> finding into a new paradigm.

---

## 1. Project in one line

Flower-based **federated learning with differential privacy (DP) for IoT**, with a working
stack: DP-SGD-style per-update noise, **secure aggregation (SecAgg)** via zero-sum integer
masks, **RDP privacy accounting**, **device-resource profiling**, and **per-client non-IID
evaluation**. Targeting a **solid IEEE venue** (IoT-J / Access / networking conference).

---

## 2. The original thesis — and why it is dead

**AdaPriv thesis:** *adaptive DP noise allocation (per-layer / per-device / per-round) beats
fixed uniform DP.*

**Verdict: not supported.** Failed across **2 datasets × 3 allocation designs**, each for a
clear structural reason (full detail in `FINDINGS.md`):

| Mechanism | Result | Root cause |
|---|---|---|
| Per-layer (reduce-only & budget-conserving) | MNIST −0.19/−0.48pp; CIFAR −3.6pp | `fc1` holds ~94% of params → **it *is* the privacy budget**; reallocation is ~zero-sum under iso-privacy |
| Per-device (fairness) | Worse mean/std/min/Gini **and** ε | `resource_score` is uncorrelated with the real disparity source (local **test** difficulty/quantity); one global model → noise can't fix a data-driven gap |
| Volatility signal on vision | CIFAR worse than MNIST | conv layers flagged "volatile → absorb noise", but they hold the critical spatial features — premise is backwards for images |

**Constraint going forward:** the user **cannot publish a negative/analysis paper**; needs a
**positive, novel** contribution.

---

## 3. Literature scans (this session) — the field is saturated

Four targeted scans, all returning **crowded** spaces. Bottom line: **DP-FL has no empty
"novel-mechanism" niche left** that fits a few-days budget.

| # | Angle scanned | Key prior art (blocks the obvious claim) | Residual gap |
|---|---|---|---|
| 1 | **DP + PEFT/LoRA in FL** | DP-LoRA (2023), FFA-LoRA (ICLR'24), FedSA-LoRA-DP (2025), DP-FedLoRA (2025), Yu et al. (ICLR'22, frozen-backbone foundation) | **ε→rank coupling** (make trainable rank a function of ε) — thin but *unclaimed* |
| 2 | **DP + personalization + fairness** | **DP-Ditto** (IEEE 2025) — extends Ditto under DP for fairness, on MNIST/FMNIST/CIFAR (our exact setup). Plus robust-clustering (2024), FedGraph-Fair, APDP-FL | Essentially **scooped**; only an incremental delta vs DP-Ditto remains |
| 3 | **DP + SecAgg + client heterogeneity** | Distributed Discrete Gaussian (Kairouz, ICML'21), Skellam (NeurIPS'21) + Skellam Mixture (VLDB'22), LAPA/HADA per-device budgets, trilemma (Chen, ICML'22) | DP-under-quantized-SecAgg is **solved**; heterogeneous budgets mature |
| 4 | **DP + quantization-precision co-design** | JoPEQ (joint DP+quant), HeteroSAg (heterogeneous quant under SecAgg), FedX (per-device quant for IoT '25), Adaptive-Quant+DP (IEEE'26) | Occupied |

**Strategic consequence:** at a **solid IEEE venue**, novelty does **not** require an empty
niche. The bar there is a **differentiated combination + a working IoT system + thorough,
honest evaluation** — exactly the class of the papers above (JoPEQ, FedX, …). That is an
achievable target with the existing harness.

---

## 4. Current direction

**An integrated private-IoT-FL system paper**, whose contribution is the combination none of
the four neighbours fully have, plus an evaluation insight we own:

1. **Integration:** SecAgg **+** DP (with the *correct* discrete-noise mechanism) **+**
   device-resource-tier behaviour, in one deployable IoT stack.
2. **Owned evaluation insight:** standard **global-test** evaluation *hides* DP's disparate
   impact (we observed std = 0 across clients); only **per-client non-IID local** evaluation
   reveals it. Report worst-client accuracy + Gini, not just mean/variance.
3. **Positive headline (uncontested by our dead results):** device-tier **precision** saves
   **communication/compute** for weak devices at fixed privacy. Fairness is *characterised*
   honestly as a secondary finding — we do **not** claim device-aware noise improves fairness
   (our own data says it does not).

> Honest caveat: this is an **integration + characterisation** contribution, not a new
> algorithm. Acceptable at the targeted venue tier; not a top-tier method paper.

---

## 5. Fix A — the one change that can improve accuracy (CONFIRMED on 2 datasets)

### The technical finding
DP is a **cost**, never an accuracy win vs non-private — so "the output is worse" is expected
and not a flaw. The relevant question is *cost vs other **private** methods*. There the code
has a real inefficiency in the **SecAgg + DP** path:

- **Current behaviour:** each client adds the **full** central noise σ (continuous Gaussian),
  then updates are summed/averaged under SecAgg. Because each of N clients adds full σ and
  SecAgg hides individuals, the **aggregate carries ≈√N more noise than necessary** for the
  claimed central-DP guarantee.
- **Fix A (distributed Skellam):** each client adds a **1/N variance share** of **discrete
  Skellam** noise in the integer/quantised domain; SecAgg sums them so the **aggregate** hits
  exactly the target noise. Skellam is closed under addition, so the sum is exactly the
  intended mechanism — and it is valid under modular integer SecAgg (unlike continuous
  Gaussian).
- **Expected effect:** same formal ε, **~√N less effective noise → higher accuracy**, with no
  trusted server. For N = 10 that is ≈3× lower noise variance on the aggregate.

### Status of Fix A — IMPLEMENTED + MEASURED (2026-06-05/06)
Implemented: `secagg-dp-mode` run-config ∈ {"", "local", "distributed"}; `skellam_noise` +
`apply_distributed_skellam_noise` in `privacy_utils.py`; `add_noise` flag on
`apply_dp_noise_per_layer` (clip-only in the Skellam path); client injects integer Skellam
after `quantize`/before mask, variance scaled 1/N for distributed. **Mechanism unit-verified:**
distributed aggregate noise = central σ (ratio 1.00); local = √N×.

**TWO PIVOTAL FINDINGS:**

- **FINDING 1 — Fix A works, but only at scale.** At **N=10** (cross-silo) both Skellam modes
  collapse to ~random (distributed ~14%, local ~19% @ ε=8) — client-level DP with N=10, full
  participation and per-layer clipping is simply too harsh; client-level DP needs the
  **cross-device** (large-N) regime. The √N→1/N separation then appears cleanly as N grows:

  | ε=8, MNIST, T=20 (peak val-acc) | N=10 | N=50 | N=100 | N=200 |
  |---|---|---|---|---|
  | no-dp ceiling (data only)            | 98.6% | 96.2% | 92.8% | 78.4% |
  | **distributed Skellam (Fix A)**      | 14.8% | 81.7% | 81.8% | 48.9% |
  | fixed-dp, no SecAgg (corrected σ)    | 15.3% | 17.1% | 24.9% | 20.8% |
  | local Skellam (per-client, SecAgg)   | 18.9% | 19.6% | 15.8% | 14.7% |
  | **DP cost** (ceiling − distributed)  | 83.8pp | 14.5pp | **11.0pp** | 29.5pp |

  Figures: `figures/crossover_peakacc_vs_N.png` (headline), `crossover_trajectories.png`
  (2×2 val-acc vs round per N), `crossover_dp_cost.png` — via
  `scripts/generate_crossover_graphs.py --eps 8.0`.

  (corrected fixed-dp N=10: 15.3% @ε8, 12.4% @ε3 — vs the buggy legacy 98.4%/97.4%.)
  **Punchline:** EVERY naive correctly-noised baseline — per-client float noise (fixed-dp,
  no SecAgg) AND per-client integer noise under SecAgg (local Skellam) — is broken (~13–25%)
  at *every* N. Only **distributed Skellam under SecAgg** reaches usable accuracy (~82%), and
  only in the N≈50–100 window. The legacy 98% fixed-dp was purely the FINDING-2 deflation bug.

  **It is a usable WINDOW, not a one-sided threshold** — governed by two competing forces:
  - *small N* → fixed central σ dominates the averaged signal → distributed collapses (N=10);
  - *large N* → **data starvation / extreme non-IID** → collapse — and this hits **no-dp too**
    (ceiling decays 98.6→78.4 with zero privacy; Dirichlet partitions fell below min size at
    N=200). The no-dp control PROVES the upper collapse is statistical, not a mechanism bug.
  - *window N≈50–100* → distributed reaches ~82%, DP cost minimized ~11pp at N=100;
    **local (naive per-client) never works at any N** (√N excess noise never clears).

  Distributed late-round drift also shrinks with N (final 47.9%→62.3%, N=50→100) — consistent
  with 1/N effective aggregate noise. SecAgg is *essential*: it is what enables the distributed
  (1/N-variance-share) noise that local cannot achieve.

  **CONFOUND to flag in the paper:** on a fixed-size dataset N and data-per-client are inversely
  coupled, so the window's *upper* edge is a data-per-client artifact, not intrinsic to client
  count (real cross-device IoT = more clients ⇒ more total data). The *lower* edge (DP-noise vs
  N — the genuine Fix A contribution) is robust. Decouple via a naturally-federated many-client
  dataset with adequate per-client data (FEMNIST/EMNIST-by-writer) before the headline claim.

- **FINDING 3 — confound resolved on FEMNIST: the crossover is MONOTONE, not a window
  (2026-06-11, analysed 2026-06-17).** Re-ran the full 4-mechanism × N∈{10,50,100,200} grid on
  **FEMNIST** (naturally federated by writer → more clients add data instead of starving it).
  The MNIST upper collapse **disappears**, exactly as the confound predicted:

  | ε=8, FEMNIST, T=20 (peak val-acc) | N=10 | N=50 | N=100 | N=200 |
  |---|---|---|---|---|
  | no-dp ceiling (data only)            | 81.2% | 66.7%¹ | 81.1% | 80.7% |
  | **distributed Skellam (Fix A)**      |  5.0% | 26.3% | 60.1% | **73.1%** |
  | fixed-dp, no SecAgg (corrected σ)    |  3.8% |  3.3% |  4.4% |  4.0% |
  | local Skellam (per-client, SecAgg)  |  7.3% |  3.4% |  4.4% |  5.1% |
  | **DP cost** (ceiling − distributed) | 76.2pp | 40.4pp | 21.0pp | **7.6pp** |

  ¹ N=50 no-dp is a slow-convergence artifact (stuck ~5% for 4 rounds, still rising at T=20 →
  66.7%; would reach ~80% with more rounds), **not** a data-starvation collapse — distributed
  N=100/200 and no-dp N=100/200 all converge cleanly to ~81%.

  **Punchline (the headline, now confound-free):** unlike MNIST (window, upper collapse at
  N=200 from data starvation), on FEMNIST distributed Skellam climbs **monotonically** toward
  the **flat** no-dp ceiling — DP cost shrinks 76→40→21→**7.6pp** as N grows, with **no upper
  collapse**. The no-dp control is flat (~81% at N=10/100/200), proving the genuine Fix A
  contribution (DP-noise vs N) is what drives the curve. Every naive baseline (per-client float
  *and* per-client integer under SecAgg) stays at random (~3–7%) at **every** N — they never
  cross. distributed N=200 is converged (peak 73.1 @ round 17, stable ~72); N=100 still drifts
  late (60.1→48.6, consistent with higher 1/N effective noise at smaller N).

  Figures: `figures/crossover_peakacc_vs_N_femnist.png` (now dataset-aware: monotone-crossover
  title + high-N shading), `crossover_trajectories_femnist.png`, `crossover_dp_cost_femnist.png`
  — via `scripts/generate_crossover_graphs.py --dataset femnist --eps 8.0`. (Generator now
  suffixes figures with `_{dataset}` so MNIST/FEMNIST coexist; detects monotone-vs-window from
  the data.) Grid: `scripts/_femnist_grid.sh` (+`_resume.sh`), logs `logs/femnist_*`, results
  `experiment_results/results_femnist_*`.

- **FINDING 2 — prior DP results were never actually private.** The legacy fixed-dp path
  divided noise by `√num_samples` (`privacy_utils.py:335`, ≈√6000 ≈ 77×), deflating effective
  σ from ~2.85 to ~0.037. **The ~98% fixed-dp numbers in `FINDINGS.md` §1/§3 ≈ no-DP**, so
  "DP costs ~1pp on MNIST" is an under-noising artifact.
  **FIXED (2026-06-06):** the `/√num_samples` factor is removed — client-level DP now adds the
  calibrated `N(0,(σ·C)²)` directly. Every legacy no-SecAgg fixed-dp result is invalid and is
  being regenerated at the correct σ (this is the "MNIST story" cleanup step).

### Code touch-points for Fix A
- `qpriviot_fl/privacy_utils.py`
  - `apply_dp_noise_per_layer` (L299) — adds continuous `torch.randn` noise; in the SecAgg path
    this should be skipped (clip only) and replaced by integer Skellam after quantisation.
  - `quantize` / `dequantize` (L376/386) — integer domain where Skellam noise is injected.
  - new: a `skellam_noise(shape, variance)` helper (Poisson − Poisson).
- `qpriviot_fl/client_app.py`
  - SecAgg branch (L312–333) — inject distributed Skellam after `quantize`, before masking;
    scale per-client variance by `1/secagg_total_clients`.
- `qpriviot_fl/server_app.py`
  - `aggregate_fit` SecAgg path (L315–334) — accounting already uses summed integers; verify the
    accountant's `accounting_noise` reflects the **aggregate** (distributed) σ, not per-client.

---

## 6. Code state (uncommitted, branch `dev/v2.4`)

**Current / live (the integration paper):**
- `qpriviot_fl/privacy_utils.py` — distributed-Skellam mechanism (`skellam_noise`,
  `apply_distributed_skellam_noise`), `add_noise` flag on `apply_dp_noise_per_layer`
  (clip-only in the Skellam path). **FIXED 2026-06-06:** removed the `/√num_samples` noise
  deflation (FINDING 2).
- `qpriviot_fl/client_app.py` — SecAgg branch injects integer Skellam after `quantize`/before
  mask; per-client variance scaled 1/N for `distributed`.
- `qpriviot_fl/server_app.py` — `secagg-dp-mode` plumbing; aggregate-σ accounting. **FIXED
  2026-06-06:** result filenames now include `nXXX` for N≠10 so scale-sweep / control runs
  no longer overwrite each other.
- `qpriviot_fl/task.py` — eval fix: per-client local non-IID test partitions (basis of the
  §4.2 disparate-impact insight; keep).
- Scripts: `scripts/_skellam_test.sh` (N=10), `_skellam_scale.sh` + `_skellam_scale_fill.sh`
  (N-sweep), `_nodp_control.sh` (no-dp ceiling curve). Logs in `logs/skellam_scale_*`,
  `logs/nodp_control_*`. Results in `experiment_results/results_mnist_*_n{50,100,200}_*`.

**Dead-thesis residue (NOT used by the current paper; revert/ignore):**
- per-layer budget-conserving allocation; device "fairness-direction" scaling
  (`client_app.py`) — made fairness worse; CIFAR/param-only/device-only result files.
- `scripts/_retest_paramonly.sh`, `_fairness_test.sh`, `_cifar_test.sh`; `generate_graphs.py`
  (MNIST-parameterised, will need re-pointing at the new result files).

Canonical config for the integration paper: SecAgg ON, seed 42, T=20, ε∈{3,8}, N swept over
{10,50,100,200}. **Two datasets:** MNIST (Dirichlet α=0.3 — fixed-size, shows the *window*) and
**FEMNIST** (writer-partitioned via `NaturalIdPartitioner`, le=5/lr=0.05 — constant per-client
data, shows the *monotone* crossover; the confound-free headline). FEMNIST result filenames do
**not** encode le/lr, so tuned runs overwrite untuned ones in place — keep that in mind on reruns.

---

## 7. Direction decision (2026-06-06) + next steps

**All "novel-mechanism" directions are dead or out of budget:** adaptive per-layer, per-device
fairness, DP-Ditto personalization, and now **DP-LoRA** (user's call: ε→rank gap too thin,
2026 preprints landing, needs edge-HW + 3 baselines). 4 lit scans + 5 dead theses ⇒ a new-
*algorithm* DP-FL paper is not achievable in a few-days budget. **Stop hunting for an empty
mechanism niche.**

**Chosen direction — characterization + deployable-system paper** anchored on the one live
positive result (Fix A's large-N crossover). Working title: *"How many devices does private
federated IoT learning actually need?"* Contributions, all already evidence-backed:
1. **Headline (positive, owned):** the **client-count crossover** — distributed-noise-under-
   SecAgg is unusable cross-silo (N≈10), usable cross-device (N≥50); naive per-client DP never
   crosses. Mechanism (Skellam) is *cited*; the IoT-regime characterization is the contribution.
2. **Methodology (owned):** global-test eval hides DP disparate impact (std=0) — only per-client
   non-IID local eval reveals it (report worst-client / Gini); + the "phantom-privacy" caution
   (FINDING 2).
3. **Negative results as design guidance** (adaptive allocation zero-sum; device-aware noise
   worsens fairness) — demoted to guardrails, not the headline.

> Optional higher-novelty lottery ticket: per-layer DP died under a *linear* (mean-σ) budget
> constraint; the correct RDP constraint is *quadratic* (Σ1/σ_j²=const), under which Fisher-
> optimal σ_j∝s_j^(−1/4) gives a provable non-zero gain. Raw Fisher already in
> `SensitivityTracker.second_moments`. Risk: fc1=94% of params caps the gain → likely marginal.
> Quick theory check + one run before committing.

### Immediate next steps
**The gating question is answered (FINDING 3, 2026-06-17): the crossover HOLDS — and is cleaner
(monotone, no upper collapse) on FEMNIST than on MNIST. We are past the go/no-go; the headline
is locked. We are now in the "build the paper" branch.** Remaining work, in priority order:

1. **DONE — N-sweep (MNIST) + all fixed-dp baselines at correct σ (FINDING 2) + FEMNIST grid
   (FINDING 3).** Both datasets' crossover figures generated
   (`figures/crossover_*_{mnist,femnist}.png`).
2. **ε-robustness:** re-run the FEMNIST grid at **ε=3** (reuse `scripts/_femnist_grid.sh` with
   `eps=3.0`) so the headline isn't single-ε. Confirms DP cost vs N at a tighter budget.
3. **Convergence cleanup (optional):** add LR-decay / early-stop so the distributed *peak*
   becomes the *final* (kills the N=100 late drift 60.1→48.6); makes the trajectory figure
   honest about deployable (final-round) accuracy.
4. **Second contribution axis:** device-tier **precision** (comm/compute savings for weak
   devices at fixed privacy) + **worst-client / Gini** reporting from the `_per_client.csv`
   files (the §4.2 disparate-impact methodology insight).
5. **Write.** Working title *"How many devices does private federated IoT learning actually
   need?"* — lead with the two-dataset crossover (MNIST window vs FEMNIST monotone), the
   phantom-privacy caution (FINDING 2), and the dead negatives demoted to design guidance.

---

## 8. Novelty pivot — 3 adversarial lit scans (2026-06-18)

The team judged the pure characterization paper too thin for the novelty they want. Three
adversarial scans (each tasked to *kill* its direction by finding scooping prior art):

| Direction | Verdict | Why |
|---|---|---|
| **Fisher-optimal per-layer DP** (quadratic RDP constraint, σ_j∝s_j^(−1/4)) | **DEAD** | Exact quadratic-constraint quarter-power closed form published Sept 2025 (arXiv:2509.04232); Fisher-weighted FL variant separately (Yan 2025, C&C P&E). Empirical gain marginal anyway (fc1=94% params). The §7 "lottery ticket" is closed. |
| **Theorize the crossover** (closed-form critical N*) | **OPEN but thin** | √N mechanism + "DP-FL needs many clients" saturated; bounds already contain N (√d/(Nε), Zhang ICML'22) → "just rearranging known bounds" reviewer risk. Apple/Pelikan 2023 already shows DP-FL viable at scale + per-layer clipping. Narrow unclaimed seam = *mechanism-conditioned* N* (naive never crosses; distributed crosses at finite N*). An *upgrade* to the existing paper, not a new one. Mid-tier (PETS/TIFS). |
| **DP + Quantum FL** | **PARTIALLY-TAKEN umbrella, H1 OPEN** | "DP+QFL" is a small 2023–25 cluster (cite, can't claim first: 2310.06973, 2509.05377, 2508.20310). DP×barren-plateau link published (2509.05377). **But unclaimed:** small-VQC-`d` → DP-cost advantage; DP-cost-vs-N in QFL; classical integer-mask SecAgg + distributed Skellam on VQC angles (QFL-SecAgg papers all use quantum-native secrecy). |

**Key synthesis — directions 1 & 2 are the SAME paper, fused by model dimension `d`.** The
crossover-theory predicts critical client count N* ∝ √d/ε. A VQC has `d≈100` angles vs the
CNN's `d≈260k`, so N* shifts left ~√(2600)≈50×. **Headline candidate:** *"The private-FL
client-count crossover is governed by model dimension — quantum FL, with tiny `d`, is
usable-private (N≈10, cross-silo) exactly where classical FL collapses."* A falsifiable,
mechanism-grounded, harness-reusing extension that spans 260k→100 params to *demonstrate* the
d-dependence. (H1 = the leftward shift; H2 = DP-noise × barren-plateau, demoted to supporting.)

**Caveats to go in clear-eyed:** (1) drop the IoT framing — no QPU on edge; reframe as
NISQ/distributed QML. (2) Mid-tier ceiling: IEEE TQE/QCE/ICASSP/QMI, not Nature/NeurIPS-main;
not "first to DP+QFL." (3) New impl: PennyLane VQC + federated loop (days); quantum sim is slow
→ small qubit counts + modest N (field-standard). (4) No published *quantum* DP-ERM √d theorem
— the N*∝√d argument is currently *classical* DP-ERM on the classical θ-angles (defensible: the
DP is on classical params), but a reviewer may probe — confirmatory check before writing.

**Status: awaiting commit decision** — (A) fused QFL paper (highest novelty, new impl, drop
IoT); (B) Direction-2-only upgrade of the existing characterization paper (safe, mid-tier);
(C) ship the characterization paper as-is. Per-client raw scan reports captured in session memory.

### 8.1 QFL scoping probe — RAN 2026-06-22 (GO signal)
Before committing, built a self-contained probe (`scripts/_qfl_probe.py`, `_qfl_dsweep.py`) that
imports the **real** project DP functions (`quantize` / `apply_distributed_skellam_noise` /
`dequantize`) so the privacy path is byte-identical to the classical crossover runs — only the
MODEL changes. Task: 3-class sklearn-digits, PennyLane VQC (`AngleEmbedding` + 2/3-layer
`StronglyEntanglingLayers`), **d=48 angles** vs the CNN's 262k. Same **σ=2.854 (ε≈8)**, T=30.
(PennyLane 0.37.0 pinned for py3.9; autoray conflict fixed.)

**Result 1 — the leftward shift is REAL and robust (2 seeds):** distributed-Skellam DP cost
(no-dp − distributed), peak val-acc:

| DP cost @ | N=5 | N=10 | N=50 |
|---|---|---|---|
| seed 42 | 7.4pp | **0.0pp** | −0.7pp |
| seed 7  | 5.2pp | **−0.7pp** | 0.7pp |

The tiny-d VQC is **usable-private at N=10 cross-silo** (~95% = no-dp ceiling) — exactly where the
260k CNN collapsed to ~14% (≈84pp DP cost). The crossover didn't vanish, it **shifted left**
from N≈50 (classical) to N≈5–10 (quantum). Naive `local` stays separated (64–75% at N=5/10),
preserving the mechanism story; separation narrows at N=50 because the task is easy + d tiny.

**Result 2 — the crossover is GOVERNED BY d (one controlled MLP sweep, N=10, same σ):**

| d | DP cost |
|---|---|
| 48 (VQC) | ~0pp |
| 275 | 0.0pp |
| 1,091 | 2.2pp |
| 4,355 | 4.4pp |
| 17,411 | 6.7pp |
| 69,635 | 7.4pp |
| 262,144 (CNN, harder 10-class task) | ~84pp |

DP cost rises monotonically with d at fixed N — direct evidence for the mechanism. (Honest: the
MLP sweep is on the easy 3-class task so its top is ~7pp; the CNN's 84pp is a *cross-task* anchor,
not a clean continuation — the within-task d-trend is the rigorous part.)

**Probe caveats (this is scoping, not the paper):** easy 3-class digits at small scale; σ reused
from the classical accountant (fair "same σ, different d" test, but the paper must calibrate σ via
a QFL accountant); N=5 lower edge is unstable; needs MNIST/FEMNIST-scale + more classes + real QFL
benchmarks. **Bottom line: the H1 hypothesis survived empirical contact → GO to commit Direction A
(fused QFL paper).** Next if committing: scale the task (MNIST 10-class via amplitude/angle
encoding), add a proper N-sweep figure (VQC vs CNN crossover on one axis), calibrate σ per QFL
rounds, then the d-governs-crossover figure as the mechanism panel.

### 8.2 SCALE-UP to MNIST 10-class — RAN 2026-06-23/24 (Direction A committed; nuanced result)
User said "go ahead" → committed Direction A and ran the full scaled experiment. Built
`scripts/_qfl_mnist.py` (federated VQC, **same** Dirichlet(α=0.3) partitioning as the CNN runs,
**byte-identical** DP path imported from `privacy_utils`, σ **calibrated via the same Opacus call**
the server uses — verified 2.854 @ε8/T=20). `scripts/_qfl_tune.py` (centralized tuner),
`scripts/generate_qfl_figures.py` (headline + mechanism figures). Data per-client = n_train//N
(re-partition per N like the CNN → replicates the MNIST data-per-client confound).

**Three engineering hurdles solved (all real, all documented in the scripts):**
1. **Barren plateau** — naive deep VQC stuck ~17-25% on 10-class MNIST. DEEPER trained WORSE.
   Fix: SHALLOW (1 entangling layer/block) + data re-uploading (PCA→NQ·blocks, one chunk/block)
   + **rich readout** ⟨Z_i⟩+⟨Z_iZ_{i+1}⟩ (2NQ−1 obs) + small linear head. NQ=8, blocks=2 →
   **d=208**. Centralized ceiling 53.5%. (PCA-24 supports 87% logreg, so features aren't the cap.)
2. **no-dp wrongly L2-clipping** (clipping is DP-only) crushed the federated ceiling to ~22%.
   Fix: no-dp = plain float FedAvg (no clip/quantize). Lifted ceiling to **47.3% @α=0.3** (52.7% IID
   → non-IID only costs ~5pp).
3. **per-client starvation** — old `cap=n_train//max(Ns)` starved N=10 to 30 samples. Fix:
   re-partition per N, cap=n_train//N (N=10→600, N=200→30).

**RESULTS (MNIST 10-class, ε=8, T=25, seed42, peak val-acc). Files: `experiment_results/
qfl_mnist_eps8.0.json`, figures `figures/qfl_crossover_vs_N_8.0.png` + `qfl_d_governs_crossover.png`.**

| peak acc | N=10 | N=50 | N=100 | N=200 |
|---|---|---|---|---|
| VQC no-dp (d=208)        | 44.7 | 42.3 | 39.2 | 32.1 |
| CNN no-dp (d=262k)       | 98.6 | 96.2 | 92.8 | 78.4 |
| VQC distributed (Fix A)  | 18.7 | 26.1 | 25.1 | 23.5 |
| CNN distributed          | 14.8 | 81.7 | 81.8 | 48.9 |
| VQC local (naive)        | 14.7 | 14.9 | 15.0 | 16.5 |
| CNN local                | 18.9 | 19.6 | 15.8 | 14.7 |
| **VQC DP cost** (pp)     | **26.0** | 16.2 | 14.1 | 8.6 |
| **CNN DP cost** (pp)     | **83.8** | 14.5 | 11.0 | 29.5 |

**d-sweep mechanism panel (controlled, 3-class digits, N=10, same σ, MLP):** DP cost rises
MONOTONE with d: d=275→0.0pp, 1091→2.2, 4355→4.4, 17411→6.7, 69635→7.4; anchors VQC d=48→0pp,
CNN d=262k→84pp. (`experiment_results/qfl_dsweep.json`.) **This controlled sweep is the cleanest,
most rigorous evidence that DP cost is governed by d.**

**HONEST ASSESSMENT (the headline is real but more NUANCED than the d=48 probe suggested):**
- **STRONG:** (a) the controlled d-sweep (same task, vary only d → monotone DP-cost-vs-d) is
  rigorous; (b) at the **cross-silo regime N=10**, the tiny-d VQC keeps DP cost to 26pp (still
  learning, 18.7%>random, rising) where the CNN COLLAPSES 84pp to ~random (14.8%); (c) naive
  `local` never crosses for either model.
- **WEAK / caveats:** (a) the VQC's absolute ceiling is only ~45% (tiny VQC on PCA-16 angles is a
  weak learner) → "usable-private" is RELATIVE (less-collapsed than CNN), NOT absolute — 18.7%
  under DP at N=10 is not an absolutely usable model; (b) VQC-vs-CNN is cross-architecture/
  cross-ceiling (98% vs 45%), so pp-DP-cost comparison is apples-to-oranges; the d-advantage is
  cleanest at N=10 and MUDDIER at N=50 (VQC 16pp ≈ CNN 14.5pp); (c) MNIST VQC shows a WINDOW
  (peaks N=50), not monotone — the data-per-client confound again.
- **Implication for the paper:** lead with the **controlled d-sweep** (mechanism) + the **N=10
  cross-silo contrast** (tiny-d retains learning where CNN dies). Do NOT overclaim "VQC gives a
  usable private model at N=10" — frame as "DP cost is governed by d; at fixed small N the
  d-advantage is large." To strengthen "usable": would need a smaller-d / fewer-class / better-
  encoded VQC with a higher ceiling, OR lean on the d-sweep as the primary result.

**Awaiting user steer on:** (1) ε=3 robustness sweep (~40min, reuse `_qfl_mnist.py --eps 3.0`);
(2) push the VQC ceiling higher (more qubits/amplitude encoding) to make "usable-private" absolute;
(3) accept the d-sweep-led framing and start writing.
