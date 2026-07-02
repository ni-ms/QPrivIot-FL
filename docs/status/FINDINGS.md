# AdaPriv — Empirical Findings (2026-06-03)

Consolidated evidence from validating the AdaPriv adaptive-DP thesis. All runs:
10 clients, Dirichlet α=0.3 non-IID, seed 42, T=20 (quick validation tier).

## Bottom line

The core thesis — *adaptive DP noise allocation (per-layer / per-device) beats
fixed uniform DP* — is **not supported** by the experiments. It failed across
**2 datasets × 3 allocation designs**, each time for a clear, structural reason.

## 1. Per-layer noise allocation (the headline "accuracy" claim)

Tested two allocation designs against fixed-DP at matched ε:

| Design | Dataset | ε | param-only vs fixed-dp |
|---|---|---|---|
| reduce-only (σ≤base) | MNIST | 3.0 / 8.0 | −0.3pp / −0.3pp (and spent *more* ε) |
| budget-conserving (iso-privacy) | MNIST | 3.0 / 8.0 | −0.19pp / −0.48pp |
| budget-conserving (iso-privacy) | CIFAR-10 | 8.0 | −3.60pp |

**Why it fails (structural):**
- **fc1 holds ~94% of parameters → it *is* the privacy budget.** Reducing its
  noise costs ε proportionally; the "reduce-only" design quietly spent more ε
  (e.g. 9.56 vs 9.16 at ε=8). Made iso-privacy via renormalization (layer-mean
  σ = base), and the accuracy edge vanished — reallocation is ~zero-sum.
- **On vision (CIFAR), it gets *worse*.** The sensitivity metric flags conv
  layers as high-sensitivity/volatile, so the allocation loads them with *more*
  noise — but conv layers hold the critical spatial features. The premise
  "volatile layer = unimportant = can absorb noise" is backwards for images.

## 2. Per-device noise allocation (the "fairness" claim)

Prerequisite fix applied first: evaluation changed from a **shared global test
set** (which gave every client identical accuracy, std=0 — fairness unmeasurable)
to **per-client local non-IID test partitions** (`task.py`). This works:
client test sizes range 228–1583 with distinct label skews.

Device scaling flipped to the fairness direction (low-resource → less noise,
`res_i/avg ∈ [0.7,1.3]`, budget-neutral on average). Result, fixed-dp vs
device-only (MNIST, final-round local-test accuracy):

| ε | config | mean | std | min (worst) | Gini | ε spent |
|---|---|---|---|---|---|---|
| 3.0 | fixed-dp | 0.953 | 0.011 | 0.934 | 0.006 | 3.75 |
| 3.0 | device-only | 0.897 | 0.041 | 0.810 | 0.025 | 5.05 |
| 8.0 | fixed-dp | 0.981 | 0.007 | 0.969 | 0.004 | 9.43 |
| 8.0 | device-only | 0.967 | 0.013 | 0.943 | 0.008 | 13.25 |

Device-aware noise is **worse on mean, std, worst-client, Gini, AND privacy.**

**Why it fails (structural):**
- **`resource_score` is uncorrelated with the real disparity source.** Per-client
  accuracy gaps come from local *test difficulty/quantity*, not from the model.
- **All clients share one global model**, so noise allocation can't fix a
  test-data-driven gap — it only injects variance → higher disparity.
- **Worst-case `min`-over-clients accounting** taxes any sub-base noise heavily
  (ε 3.75→5.05, 9.43→13.25).

## 3. Dataset headroom

| Dataset (T=20) | no-dp | fixed-dp | DP cost |
|---|---|---|---|
| MNIST | 98.6% | 97–98% | ~1pp |
| CIFAR-10 | 63.2% | 58.4% | 4.9pp (undertrained at T=20) |

MNIST has essentially no headroom for adaptation to exploit. CIFAR has more, but
even there the per-layer claim was negative. (CIFAR T=20 is undertrained;
no-dp would reach ~75–80% at T=100 — not run, per decision to stop testing.)

## Implications for paper framing

The adaptive-allocation-beats-uniform claim is not viable as stated. Options
discussed: (a) reframe as a systems/framework contribution (Flower + SecAgg +
RDP accounting IoT harness with configurable adaptive DP); (b) honest negative /
analysis result; (c) redesign the core adaptive criterion. Decision deferred.

## Code state (uncommitted, this session)

- `qpriviot_fl/task.py` — **eval fix** (per-client local non-IID test partitions).
  This is arguably a genuine correctness improvement, but it changes the
  experimental setup (global → federated-local evaluation). Decide whether to keep.
- `qpriviot_fl/privacy_utils.py` — budget-conserving per-layer allocation
  (iso-privacy renormalization). More honest than the prior reduce-only design.
- `qpriviot_fl/client_app.py` — **device scaling flipped** to fairness direction.
  This was experimental and made fairness *worse*; consider reverting.
- `scripts/generate_graphs.py` — rewritten for MNIST, parameterized, robust.
- Temp run scripts: `scripts/_retest_paramonly.sh`, `_fairness_test.sh`,
  `_cifar_test.sh`.
