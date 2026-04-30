# QPrivIoT-FL Code Review for Peer-Reviewed Artifact

**Reviewer:** Senior Research Software Engineer (Python)
**Scope:** `qpriviot_fl/` package + `plot.py`
**Goal:** Make the codebase robust, reproducible, and defensible for a peer-reviewed artifact.

Files inspected:
- `qpriviot_fl/__init__.py`, `config.py`, `task.py`, `device_profile.py`,
  `client_app.py`, `server_app.py`, `privacy_utils.py`
- `plot.py`
- `pyproject.toml`, `README.md`, `run_*.sh`

---

## 1. Organizational Structure

### 1.1 Findings

| # | File:Line | Issue | Severity |
|---|-----------|-------|----------|
| 1.1 | `task.py:73` | `load_data()` mixes three concerns: synthetic IoT generation, HF dataset download/caching, and DataLoader construction. The `_fds_cache` module-global is a hidden side-effect. | High |
| 1.2 | `task.py:12-59` | Three model classes (`Net`, `FemnistNet`, `IotModel`) live next to data-loading. A `models/` subpackage would scale better. | Med |
| 1.3 | `client_app.py:37-251` | `fit()` is ~210 lines; it interleaves resource gating, DP-mode selection, training, gradient-norm bookkeeping, SecAgg masking, and metrics. PEP-8 readable size is ~50 lines. | High |
| 1.4 | `server_app.py:179-334` | `aggregate_fit()` does (a) resource bookkeeping, (b) SecAgg dequantization, (c) weighted FedAvg, (d) RDP accounting, (e) sensitivity tracking, (f) logging, (g) checkpointing — all in one method. | High |
| 1.5 | `server_app.py:110` | `import random` inside the function body. Move to top. | Low |
| 1.6 | `server_app.py:312` | `if hasattr(self, '_get_round_epsilon')` defensive check on a method that always exists on `self` — dead code. | Low |
| 1.7 | `plot.py:25` | `seaborn` is imported and a style is set, but `seaborn` is not declared in `pyproject.toml`. Reproducibility breaks for fresh installs. | High |
| 1.8 | `task.py:9` | `from datasets import load_dataset` — the `datasets` HF package is not declared in `pyproject.toml` either (it ships transitively via `flwr-datasets`, but a direct dep should be explicit). | Med |
| 1.9 | `client_app.py:21` | `DEVICE_MAP` is a module-level magic constant duplicating keys from `config.resource.device_distribution`. Drift risk if config changes. | Med |
| 1.10 | `client_app.py:81, 258` | Device-selection ladder (`cuda → mps → cpu`) is duplicated. Extract `_pick_device()`. | Low |
| 1.11 | `task.py:71` | `_fds_cache` is a process-global keyed on `(name, num_partitions)` — fine for simulation, but undocumented and complicates testing (state leaks between tests). | Med |
| 1.12 | `qpriviot_fl/test/` | Directory exists but contains only docs and two non-test modules (`metrics.py`, `privacy.py`). No `pytest`-discoverable tests. | High |
| 1.13 | All files | No type hints on public APIs of `task.py` (e.g., `train`, `test`, `load_data`). Inconsistent with `privacy_utils.py` which has thorough hints. | Low |
| 1.14 | `client_app.py:248-251` | `except Exception as e: ... raise e` — re-raising loses traceback chain compared to bare `raise`. | Low |
| 1.15 | All files | Print-based logging (`print(...)`). For peer review you want structured logs; switch to `logging.getLogger(__name__)`. | Med |

### 1.2 Recommended Layout

```
qpriviot_fl/
├── __init__.py
├── config.py                # dataclasses (already good)
├── data/
│   ├── __init__.py
│   ├── loaders.py           # load_data + dataset routing
│   └── synthetic_iot.py     # IoT toy dataset only
├── models/
│   ├── __init__.py
│   ├── cifar.py             # Net
│   ├── femnist.py           # FemnistNet
│   └── iot.py               # IotModel
├── training/
│   ├── train.py             # train()
│   └── eval.py              # test()
├── privacy/
│   ├── accountant.py        # RenyiPrivacyAccountant
│   ├── adaptive_noise.py    # allocate_adaptive_noise + apply_dp_noise_per_layer
│   ├── sensitivity.py       # SensitivityTracker
│   └── secagg.py            # quantize / dequantize / masks
├── federation/
│   ├── client_app.py        # thin: orchestrates calls
│   └── server_app.py        # strategy in pieces (configure_fit, aggregate, save)
├── device_profile.py
└── tests/
    ├── unit/
    └── integration/
plot.py  → split: io.py (load_experiments / aggregate_seeds) + plots/*.py
```

Refactor `fit()` and `aggregate_fit()` into named private helpers
(`_resource_gate`, `_select_dp_mode`, `_apply_secagg_mask`, `_aggregate_secagg`,
`_aggregate_weighted`, `_update_accountant`, `_log_round`).

### 1.3 PEP 8 / Style Cleanups

- Line lengths exceed 100 chars in `client_app.py:78, 137, 154, 166, 244` and `server_app.py:142, 273-274`. Run `black -l 100` and `ruff check`.
- `from qpriviot_fl.config import DEFAULT_CONFIG as CONFIG` at `client_app.py:11` — fine, but inconsistent with `device_profile.py:4` which imports the same name without alias.
- Trailing whitespace and inconsistent blank lines around dataclasses; `ruff format` fixes both.
- Add a top-level `Makefile` or `tox.ini` with `lint`, `test`, `format` targets.

---

## 2. Scientific Validity

### 2.1 Critical issues (block publication if unfixed)

| # | File:Line | Issue | Why it matters |
|---|-----------|-------|----------------|
| **S1** | `task.py:98` | `val_set = load_dataset(hub_dataset_name, split="test")` — every client (and every call) re-downloads / loads the **full** test set, but it's then used only inside the client. **No client/server split for evaluation.** Combined with `client_app.py:206`, every client evaluates on the *global* test set, then those numbers are averaged. This is **train-on-train, eval-on-test**, fine in itself, but: (i) the val-loss reported is the same set across clients → no statistical independence, (ii) the val-loader at `task.py:116` shuffles nothing, but transforms run per-call so caching is wasted. | Reproducibility / efficiency |
| **S2** | `task.py:77-83` | IoT dataset is `torch.randn(...)` with `np.random.seed(partition_id)` — but `torch.randn` is **not** seeded by `np.random.seed`. Train and val tensors are non-deterministic. | Reproducibility — this is the main IoT data path. |
| **S3** | `device_profile.py:26-33` | `random.seed(partition_id)` is set, then four `random.uniform` draws follow. With `partition_id` fixed, the device profile is **deterministic per partition for the entire run**. The comment at line 21 says "Remove fixed seed to allow dynamic battery/network status" but the seed *is* applied when `partition_id is not None`, which is always true for a Flower client. Battery/network are static. | Scientific claim that the system reacts to dynamic device state is unsupported. |
| **S4** | `privacy_utils.py:109` | RDP for sub-sampled Gaussian uses `α·q²·steps / (2σ²)`. This is the **simple upper bound**, valid only for `q ≤ 1/(5σ)` (Mironov & Talwar 2019, Wang+ 2019). With `target_epsilon=12` and `num_rounds=50`, `σ ≈ 0.66`, and `q = fraction_fit · num_clients / num_clients ≈ 0.8`, so `q ≫ 1/(5σ)`. The bound **is not valid** in your operating regime. Use Opacus' `RDPAccountant` or `prv_accountant`. | The headline ε numbers are not formally defensible. |
| **S5** | `client_app.py:143` | `fixed_noise_multiplier = sqrt(2·ln(1.25/1e-5)) / max(epsilon_t, 1e-6)`. This is the analytic-Gaussian formula for **a single mechanism**, then composed implicitly by `_get_round_epsilon`'s "divide by sqrt(T) · 2" heuristic on `server_app.py:72`. The 2.0 multiplier is described as a "calibration constant" — it is not. The privacy accounting is decoupled from the noise calibration, so the reported `total_epsilon` is incoherent with the noise actually injected. | Privacy claim is unsound. |
| **S6** | `privacy_utils.py:307-309` | `noise_std = (sigma * clip_norm) / sqrt(num_samples)`. In central DP-FL the noise is added **once per client update with std `σ·C`**, not divided by `√n`. Dividing by `√n` (where `n` is local samples) makes the noise vanish for clients with many samples — the common reason FL papers under-report ε. | Wrong noise scale → reported privacy is over-stated. |
| **S7** | `server_app.py:241-258` | Resource-weighted aggregation multiplies by `resource_score` *and* uses non-uniform sampling at lines 90-113. The sampling weight is unaccounted for in the privacy amplification analysis (which assumes uniform sub-sampling). | Sampling-rate-by-subsampling amplification claim is invalid. |
| **S8** | `client_app.py:204-206` | `evaluate(eval_params, ...)` is called **inside** `fit()` *before* DP noise is added (good) but uses the same `val_loader` that every other client uses. The same numbers are then averaged in `aggregate_fit()` (line 304). This double-counts the same evaluations. | Inflated/redundant metrics. |
| **S9** | `task.py:121` | `train()` clips per-step gradients with `clip_grad_norm_(..., max_norm=5.0)`. This is **batch-level** clipping. DP-SGD requires **per-example** clipping. Adding noise to a non-DP gradient does not give DP. | Privacy guarantee invalid for the "fixed-DP" baseline if your story relies on per-sample clipping. State explicitly that you use **central DP at the update level**, not DP-SGD. |
| **S10** | `client_app.py:46-47, 89-92` | `avg_sensitivity = 0.0` is set twice (lines 45 & 89) — second assignment is unreachable for the success path because line 89 is on a path where lines 45-46 already ran. Cosmetic, but indicative of error-handling drift. | Low |

### 2.2 Other issues

- **Hardcoded hyperparameters that should live in `config.py` / `pyproject.toml`:**
  - `client_app.py:111-112` — `sigma_i_factor` clip range `[1.0, 2.0]`.
  - `client_app.py:133, 150` — fine-tuning noise scale `0.8`.
  - `client_app.py:146` — `max_norm=5.0` (also `task.py:146`).
  - `client_app.py:160` — fixed_clip_norm `5.0`.
  - `device_profile.py:36-37` — `alpha_batt=0.7, alpha_net=0.8`.
  - `server_app.py:72` — `* 2.0` calibration constant.
  - `server_app.py:103` — `0.7 / 0.3` high-resource / random sampling split.
  - `server_app.py:147` — `* 0.5` LR decay after 25% of rounds.
  - `privacy_utils.py:172` — `0.9 / 0.1` EMA factor.
  - `privacy_utils.py:265` — `s_j_clip` range `[0.5, 2.0]`.
  - `privacy_utils.py:319-320` — `min_norm=0.01, max_norm=10.0`.

  Centralize in a `HyperparameterConfig` dataclass and surface in `pyproject.toml`'s `[tool.flwr.app.config]` so all numbers in the paper trace to a single config.

- **NaN handling:**
  - `aggregate_fit` (`server_app.py:300-307`): `np.mean([...])` over empty list returns `nan` with a warning. Guard with `if values: ...`.
  - `privacy_utils.py:209-213`: division by `impact_mean`; if all gradients are zero (round 1 with frozen layers), produces `1.0` fallback — OK, but no log.

- **Improper scaling / leakage candidates:**
  - `task.py:101-103` — Normalize uses `(0.5, 0.5, 0.5)` for CIFAR-10 mean/std. The paper-standard mean/std is `(0.4914, 0.4822, 0.4465)` and `(0.2470, 0.2435, 0.2616)`. With the wrong normalization, accuracy will be lower and not comparable to literature.
  - No train/val split inside the partition: validation is the *test* set itself. Either split each client's partition 80/10/10 or rename "val" → "test".

- **Determinism / reproducibility:**
  - No global `seed_everything()` — `torch`, `numpy`, `random` are seeded inconsistently.
  - `DataLoader(... num_workers=2)` (`task.py:112`) without `worker_init_fn` → non-deterministic worker RNG.
  - `secrets.randbits(32)` for `secagg_seed` (`server_app.py:56`) — non-reproducible by design; for the paper artifact accept a `--seed` CLI flag.
  - `quick_proof.sh`/`run_experiments.sh` should pin seeds and a single `git rev-parse HEAD` into the results filename.

- **Privacy budget enforcement:**
  - `server_app.py:272-275` halts training when the budget is exceeded but does not save the *last valid* checkpoint separately. The `_save` call afterwards never executes for that round. Either save before the budget check or persist before halting.

---

## 3. Efficiency

| # | File:Line | Issue | Fix |
|---|-----------|-------|-----|
| E1 | `task.py:98` | `load_dataset(..., split="test")` runs **every fit/evaluate call on every client**. On 10 clients × 50 rounds × 2 loads = 1000 loads of the full HF test set. | Cache by dataset name in `_fds_cache` or move val to the server. |
| E2 | `client_app.py:80, 256` | `load_data(...)` re-instantiates the partitioner each call after the first (fine — cached). But the **transform** is re-applied each call. For HF datasets, `with_transform` is cheap; the bottleneck is `set_format('torch')` followed by per-image PIL→tensor conversion in workers. Pre-tokenize / pre-tensorize once if memory allows. | Pre-cache tensors in `/tmp` on first call. |
| E3 | `client_app.py:178-183` | Loop over `model.named_parameters()` to compute deltas, then again at lines 191-199 to convert to NumPy. Two full passes. | One pass: build `deltas` and `numpy_initial` together. |
| E4 | `privacy_utils.py:386-397` | `generate_zero_sum_masks` builds masks in Python list comprehensions and calls `sum(random_masks)` (a Python `sum` over NumPy arrays). For large layers this is slow. | Use `np.stack(masks).sum(axis=0)` or `rng.integers` once for the whole tensor. |
| E5 | `server_app.py:286-292` | Tight loop `torch.from_numpy(...)` for each layer to compute the norm — converts CPU NumPy to torch only to call `torch.norm`. | `np.linalg.norm(arr)` is identical and avoids the round-trip. |
| E6 | `server_app.py:300-307` | Five separate list comprehensions over `results`. Build one `pandas.DataFrame` (or one pass). | Single loop or `pd.DataFrame.from_records(...)`. |
| E7 | `client_app.py:155` | `clamped_sensitivities = {k: max(v, 1e-6) for k, v in sensitivities.items()}` is fine, but the JSON dump in line 166 then has the server send the same string to every client. | Compute once per round, not per client. |
| E8 | `task.py:130-152` | `total_loss += loss.item()` synchronizes GPU→CPU each step. For large models this is the dominant cost. | Accumulate as a tensor and `.item()` once at end. |
| E9 | `device_profile.py:25` | `random.choices(..., weights=...)` per call is fine; no fix needed. | — |
| E10 | `plot.py:131-148` | `defaultdict(lambda: defaultdict(list))` followed by `np.mean/std` per key — quadratic in seeds × metrics × rounds. | Build a 3-D NumPy array once per (config, metric). |

---

## 4. Testing Protocol

### 4.1 Unit tests (edge cases first)

#### `qpriviot_fl/privacy_utils.py`

- `RenyiPrivacyAccountant`
  - `add_round(noise_multiplier=0.0, ...)` returns `inf` (verify).
  - `get_total_epsilon()` on empty history → `0.0`.
  - Idempotency: two `add_round(σ, q, 1)` calls equal one `add_round(σ, q, 2)`.
  - Monotonicity: ε(t) is non-decreasing.
  - Convergence to literature: σ=1.0, q=0.01, T=10000 → ε ≈ ?, compare to Opacus.

- `allocate_adaptive_noise`
  - Empty `sensitivities` → `({}, {}, base_sigma)`.
  - All `s_j == 1.0` → all `noise_multipliers == base_sigma`.
  - Non-positive `s_j` (e.g. `0`, `-1`, `inf`) → fallback to `1.0`, no crash.
  - `target_epsilon → 0` does **not** raise (clamp at `1e-6`).

- `apply_dp_noise_per_layer`
  - `sigma=0` and `clip_norm=∞` → output equals input (no change).
  - With deterministic RNG (`torch.manual_seed`), output is reproducible.
  - `delta` larger than clip → `||delta_clipped|| ≤ clip_norm` (within float).
  - `num_samples=0` → no division-by-zero (guarded by `max(num_samples,1)`).

- `SensitivityTracker`
  - First `update` with single layer → `get_sensitivities()` returns `{layer: 1.0}` (post-normalization).
  - All-zero gradients across rounds → all layers `0.1` (the floor).
  - Window slides correctly past `window_size`.

- `quantize` / `dequantize`
  - Round-trip: `dequantize(quantize(x)) ≈ x` within `clip_range/range_max` precision.
  - Out-of-range values get clipped (no overflow).

- `generate_zero_sum_masks`
  - Sum of all client masks per layer is exactly `0` (int64).
  - Shape preservation per layer.
  - Same seed → same masks.

#### `qpriviot_fl/task.py`

- `make_model("cifar10" | "femnist" | "iot" | "unknown")` returns the right class; default falls through to `Net` (document this).
- `train(model, loader, epochs=0, ...)` returns `0.0` (no division by zero).
- `test(model, empty_loader, ...)` returns `(0.0, 0.0)` (already guarded — verify).
- `load_data("iot", ...)` returns two `DataLoader`s with the documented shapes (500, 10) / (100, 10).

#### `qpriviot_fl/device_profile.py`

- Same `partition_id` ⇒ same profile (current behavior — call out as a limitation).
- `cap_i ∈ [0.05, 1.0]`, `mem_i ∈ [0.1, 1.0]`, `batt_i ∈ [0.1, 1.0]`, `net_i ∈ [0.2, 1.0]`.
- `readiness_score == min(cap, mem, batt·0.7, net·0.8)`.

#### `qpriviot_fl/server_app.py` (strategy)

- `_get_round_epsilon(t=1)` ≤ `target_epsilon`.
- Cosine schedule starts at `epsilon_init` and ends at `epsilon_final` (within tol).
- `aggregate_fit([], [])` returns `(None, {})`.
- Budget breach: feed enough fake `results` to exceed ε → returns `{"budget_exceeded": True}`.

### 4.2 Integration tests

1. **Single-round end-to-end with synthetic IoT.** `dataset=iot`, `num-server-rounds=1`, 3 clients. Asserts: server log file is created, `global_model.npz` exists, no exceptions, ε after one round equals analytical ε for σ used.
2. **Three-mode regression.** Run 3 rounds with `use-dp=false`, `use-dp=true`, `use-adaptive-dp=true` on IoT. Assert all three produce a `results_*.json` with exactly 3 entries. Compare final accuracies are within ±5pp of a checked-in reference (golden file).
3. **SecAgg correctness.** With `use-secagg=true` and `use-dp=false`, assert that the server-aggregated update equals the unmasked weighted average within float tolerance (mocking 2 clients).
4. **Determinism.** Same seed twice → byte-identical `global_model.npz`.
5. **Plot pipeline.** Generate a small synthetic `experiment_results/` directory and run `plot.py`. Assert all 7 PNGs exist and have non-zero size.

### 4.3 Sanity checks (toy datasets / expected ranges)

- **Toy MLP, no DP, MNIST 1k subset, 5 rounds**: val accuracy > 80%.
- **Same with `use-dp=true`, ε=8**: val accuracy ≥ no-DP - 10pp.
- **Same with ε=0.5**: accuracy near random (~10%).
- **Privacy budget self-test**: `accountant.get_total_epsilon()` after `T` rounds with σ=1, q=0.01 must equal Opacus' RDP accountant within 0.05.
- **Zero-sum masks**: for any seed, `sum(masks_per_client[i] for i)` per layer must be exactly zero; assert across 100 random seeds.
- **Quantization round-trip error**: `||x - dequantize(quantize(x))||_∞ < 2 · clip_range / range_max`.

---

## 5. Execution & Results

### 5.1 Test harness layout (create these files)

```
tests/
├── conftest.py
├── unit/
│   ├── test_privacy_accountant.py
│   ├── test_adaptive_noise.py
│   ├── test_sensitivity.py
│   ├── test_secagg.py
│   ├── test_task.py
│   └── test_device_profile.py
└── integration/
    ├── test_pipeline_iot.py
    ├── test_dp_modes.py
    └── test_plotting.py
```

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov", "pytest-xdist", "pytest-benchmark", "ruff", "black", "opacus"]

[tool.pytest.ini_options]
addopts = "-ra --strict-markers --tb=short"
testpaths = ["tests"]
markers = [
  "slow: integration tests (>30s)",
  "gpu: requires CUDA",
]
```

### 5.2 Template — `tests/conftest.py`

```python
import os, random, numpy as np, torch, pytest

@pytest.fixture(autouse=True)
def _deterministic():
    seed = 1337
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # set True if your CUDA build supports it
    os.environ["PYTHONHASHSEED"] = str(seed)
    yield

@pytest.fixture
def tiny_cnn():
    from qpriviot_fl.task import make_model
    return make_model("cifar10")
```

### 5.3 Template — `tests/unit/test_privacy_accountant.py`

```python
import math, pytest
from qpriviot_fl.privacy_utils import RenyiPrivacyAccountant

def test_empty_history_zero_epsilon():
    assert RenyiPrivacyAccountant().get_total_epsilon() == 0.0

def test_zero_noise_is_infinite():
    acc = RenyiPrivacyAccountant()
    acc.add_round(noise_multiplier=0.0, sampling_rate=0.1)
    assert math.isinf(acc.get_total_epsilon())

def test_monotonic_growth():
    acc = RenyiPrivacyAccountant()
    eps = []
    for _ in range(5):
        acc.add_round(noise_multiplier=1.0, sampling_rate=0.05)
        eps.append(acc.get_total_epsilon())
    assert eps == sorted(eps)

def test_steps_equiv_to_repeated_calls():
    a = RenyiPrivacyAccountant(); b = RenyiPrivacyAccountant()
    a.add_round(1.0, 0.05, steps=5)
    for _ in range(5): b.add_round(1.0, 0.05, steps=1)
    assert a.get_total_epsilon() == pytest.approx(b.get_total_epsilon(), rel=1e-9)

@pytest.mark.parametrize("sigma,q,T", [(1.0, 0.01, 100), (2.0, 0.05, 50)])
def test_against_opacus(sigma, q, T):
    opacus = pytest.importorskip("opacus.accountants").RDPAccountant()
    for _ in range(T): opacus.step(noise_multiplier=sigma, sample_rate=q)
    eps_opacus = opacus.get_epsilon(delta=1e-5)
    acc = RenyiPrivacyAccountant(target_delta=1e-5)
    for _ in range(T): acc.add_round(sigma, q)
    assert acc.get_total_epsilon() == pytest.approx(eps_opacus, rel=0.10)
```

### 5.4 Template — `tests/unit/test_secagg.py`

```python
import numpy as np, pytest
from qpriviot_fl.privacy_utils import quantize, dequantize, generate_zero_sum_masks

def test_quantize_roundtrip():
    x = [np.random.randn(4, 4).astype(np.float32) for _ in range(3)]
    y = dequantize(quantize(x, 1.0, 1_000_000), 1.0, 1_000_000)
    for a, b in zip(x, y):
        assert np.max(np.abs(np.clip(a, -1, 1) - b)) < 2e-6

@pytest.mark.parametrize("n_clients", [2, 5, 10])
def test_zero_sum_masks(n_clients):
    shapes = [(3, 3), (10,), (2, 4, 4)]
    masks = generate_zero_sum_masks(shapes, n_clients, seed=42)
    for layer in range(len(shapes)):
        s = sum(masks[c][layer] for c in range(n_clients))
        assert np.all(s == 0)
```

### 5.5 Template — `tests/integration/test_pipeline_iot.py`

```python
import json, os, subprocess, sys, pathlib

def test_one_round_iot(tmp_path):
    env = {**os.environ, "QPRIVIOT_RESULTS_DIR": str(tmp_path)}
    cmd = [sys.executable, "-m", "flwr", "run", ".",
           "--run-config",
           "dataset=iot num-server-rounds=1 use-dp=false use-adaptive-dp=false"]
    proc = subprocess.run(cmd, cwd=pathlib.Path(__file__).parents[2],
                          env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    out = pathlib.Path("results_no_dp.json")
    assert out.exists()
    data = json.loads(out.read_text())
    assert len(data["rounds"]) == 1
```

### 5.6 Terminal commands

Install dev deps and run:

```bash
# from repo root
pip install -e '.[dev]'

# 1) Lint + format check
ruff check qpriviot_fl tests
black --check -l 100 qpriviot_fl tests

# 2) Unit tests with coverage, machine-readable for the appendix
mkdir -p test_artifacts
pytest tests/unit \
  -v \
  --maxfail=1 \
  --durations=20 \
  --cov=qpriviot_fl \
  --cov-report=term-missing \
  --cov-report=xml:test_artifacts/coverage.xml \
  --junitxml=test_artifacts/junit_unit.xml \
  | tee test_artifacts/unit_tests.log

# 3) Integration tests (slow)
pytest tests/integration -v -m "not gpu" \
  --junitxml=test_artifacts/junit_integration.xml \
  | tee test_artifacts/integration_tests.log

# 4) Convert junit → CSV for the paper appendix
python scripts/junit_to_csv.py test_artifacts/junit_unit.xml \
  test_artifacts/junit_integration.xml \
  > test_artifacts/test_results.csv

# 5) Reproducibility smoke run
SEED=1337 bash run_mini_suite.sh 2>&1 | tee test_artifacts/mini_suite.log
sha256sum global_model.npz >> test_artifacts/mini_suite.log
```

### 5.7 `scripts/junit_to_csv.py` (CSV exporter for appendix)

```python
"""Usage: python junit_to_csv.py junit1.xml junit2.xml > results.csv"""
import sys, csv, xml.etree.ElementTree as ET

w = csv.writer(sys.stdout)
w.writerow(["suite", "classname", "name", "status", "time_s", "message"])
for path in sys.argv[1:]:
    root = ET.parse(path).getroot()
    suites = root.findall("testsuite") or [root]
    for s in suites:
        for tc in s.findall("testcase"):
            status = "passed"; msg = ""
            if tc.find("failure") is not None:
                status, msg = "failed", (tc.find("failure").get("message") or "").splitlines()[0]
            elif tc.find("error") is not None:
                status, msg = "error", (tc.find("error").get("message") or "").splitlines()[0]
            elif tc.find("skipped") is not None:
                status = "skipped"
            w.writerow([s.get("name", path), tc.get("classname"), tc.get("name"),
                        status, tc.get("time", "0"), msg])
```

### 5.8 Reproducibility checklist for the artifact appendix

- [ ] `pyproject.toml` pins `torch==2.7.1` (already done) — also pin `flwr`, `flwr-datasets`, `numpy`, `seaborn`.
- [ ] Single `--seed` CLI flag wired through `secagg_seed`, `torch.manual_seed`, `np.random.seed`, `random.seed`, and a `worker_init_fn` for DataLoader.
- [ ] `git rev-parse HEAD` recorded into every `results_*.json`.
- [ ] `python -V`, `pip freeze`, `nvidia-smi` (if GPU) captured in `test_artifacts/env.txt`.
- [ ] Each result file references the *exact* config it ran with (full `run_config` dict).
- [ ] Privacy accountant validation against Opacus included as a CI test.
- [ ] CIFAR-10 normalization corrected to dataset-statistics values; re-run baselines.
- [ ] `train()` clarified as central-update DP, not DP-SGD, **or** switched to per-sample clipping via Opacus' `PrivacyEngine`.

---

## Priority-Ordered Action List

**Must-fix before submission**
1. **S4 / S5 / S6** — privacy accounting and noise calibration are not internally consistent. Replace `_compute_rdp_gaussian` with Opacus' `RDPAccountant` and remove the ad-hoc `÷ √(num_samples)` and `× 2.0` constants.
2. **S9** — disclose central vs per-sample DP, or switch to Opacus.
3. **S2** — fix IoT data RNG (`torch.manual_seed(partition_id)`).
4. **1.7 / 1.8** — declare `seaborn`, `datasets` in `pyproject.toml`.
5. **1.12** — create `tests/` and add at minimum the privacy accountant + secagg unit tests.

**Should-fix**
6. **1.3 / 1.4** — split `fit()` and `aggregate_fit()` into helpers.
7. **2.2** — externalize hardcoded hyperparameters.
8. CIFAR-10 normalization constants.
9. Determinism: global seed plumbing + `worker_init_fn`.

**Nice-to-have**
10. Layout refactor in §1.2.
11. Replace `print` with `logging`.
12. Efficiency wins E1, E5, E8.