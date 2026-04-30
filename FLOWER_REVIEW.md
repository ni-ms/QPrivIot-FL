# QPrivIoT-FL — Deep Flower / FL Engineering Review

**Reviewer:** Senior Research Engineer, Distributed Systems & Federated Learning (Flower)
**Framework targets:** `flwr[simulation] >= 1.22.0`, `flwr-datasets >= 0.5.0`, PyTorch 2.7
**Scope:** `qpriviot_fl/server_app.py`, `client_app.py`, `task.py`, `device_profile.py`,
`privacy_utils.py`, `pyproject.toml`, the `run_*.sh` scripts.

This review is complementary to `CODE_REVIEW.md` and focuses on Flower-specific
correctness, simulation efficiency, and FL experimental protocol.

---

## TL;DR — Critical Fixes (must fix before submission)

1. **Sampling bug** — `client_manager.sample(num_clients=self.fraction_fit, ...)`
   (`server_app.py:115-123`) passes a *fraction* (0.8) where an *integer count* is
   required. Flower's `ClientManager.sample` does `random.sample(..., k=num_clients)`
   which silently uses `int(0.8) == 0` or returns the `min_num_clients` floor —
   the experimental fraction-fit is **not honored** in the SecAgg / non-AdaPriv
   branches.
2. **No backend resource allocation.** `[tool.flwr.federations.local-simulation]` only
   sets `options.num-supernodes = 10`. With Ray, every virtual SuperNode tries to
   share the same GPU/CPU. Add `options.backend.client-resources`.
3. **Partitioner is `IidPartitioner` everywhere** (`task.py:92`). The paper claims
   relevance to heterogeneous IoT — IID partitioning **does not support that
   claim**. Switch to `DirichletPartitioner(num_partitions=N, alpha=0.3, ...)`.
4. **Federated evaluation is short-circuited.** `client_app.py:206` calls
   `self.evaluate(...)` *inside* `fit()` instead of letting Flower dispatch
   evaluation through `configure_evaluate` / `aggregate_evaluate`. This:
   (a) doubles the work, (b) bypasses `fraction_evaluate`, (c) reports
   accuracy on the same global test set duplicated across clients.
5. **Parameters are propagated via `named_parameters()` only**
   (`client_app.py:271-277`, `server_app.py:374`). Buffers (BatchNorm running
   stats, etc.) and any non-trainable state are silently dropped. Not a bug for
   the current `Net`/`FemnistNet` (no BN), but it is a landmine for any
   reviewer-suggested model swap.
6. **`fit_metrics_aggregation_fn` is never set.** The strategy returns
   `agg_params, {}` (`server_app.py:334`). Flower's metrics history is empty;
   only the side-channel `experiments_log` JSON has results. Set
   `fit_metrics_aggregation_fn=weighted_avg_metrics` and `evaluate_metrics_aggregation_fn`
   so reviewers can re-run and read metrics from Flower's standard `History` object.
7. **`secrets.randbits(32)` for SecAgg seed** (`server_app.py:56`) is non-reproducible
   by design. For the artifact, derive from a `--seed` flag.
8. **No `torch.cuda.empty_cache()` / no model reuse.** A fresh `make_model(...)`
   is built and pushed to device on every `fit` *and* `evaluate` call. With 10
   SuperNodes × 50 rounds × 2 calls = 1000 model allocations per run.

---

## 1. Architectural Organization (Flower-specific)

### 1.1 ClientApp / ServerApp pattern

You are on the new `flwr.clientapp.ClientApp(client_fn=...)` /
`flwr.server.ServerApp(server_fn=...)` API. Good. Two issues:

- `client_app.py:280` returns `QPrivIoTClient(context).to_client()`. Correct call,
  but the `__init__` does a lot of work (loads `device_profile`, etc.). In
  simulation, `client_fn` is invoked **once per round per virtual SuperNode**;
  every round you re-profile the device, re-read `node_config`, etc. Move
  one-shot work into a class-level cache keyed on `partition_id`, or accept that
  the `device_profile` is re-randomized every round (which contradicts the
  static-seed comment on `device_profile.py:21`).
- `server_app.py:346` — `server_fn` builds the strategy fresh every run. That is
  correct; just be aware that `np.savez(self.model_file, ...)` writes to
  `os.path.abspath(...)` which is the **CWD of the simulation runner** — for
  Ray-based simulation, the CWD inside actors can differ. Use
  `context.node_config.get("output_dir", ...)` or a path passed via
  `run_config`, not `os.path.abspath` from the strategy's working directory.

### 1.2 `NumPyClient` parameter handling

| # | File:Line | Issue | Fix |
|---|-----------|-------|-----|
| C1 | `client_app.py:269-273` | `_set_parameters` uses `named_parameters()` and `param.data = tensor.view(...)`. This silently drops buffers and bypasses dtype/device validation. | Use `state_dict` round-trip: build an `OrderedDict` zipping `model.state_dict().keys()` with the incoming arrays (cast via `torch.as_tensor`) and call `model.load_state_dict(..., strict=True)`. |
| C2 | `client_app.py:275-277` | `_get_parameters` returns trainable parameters only — server stores trainables only. Combined with C1, BN/LN buffers never sync across the federation. | Same — switch both ends to `state_dict()` / `load_state_dict()`. |
| C3 | `client_app.py:80, 256` | `load_data(...)` is called in *both* `fit` and `evaluate`. With the framework's standard pipeline, this is two HF dataset loads per round per client. | Cache `(train_loader, val_loader)` as instance attributes initialized lazily in `fit`. |
| C4 | `client_app.py:81, 258` | Device picking duplicated. | Helper `_pick_device()` at module scope. |
| C5 | `client_app.py:206-207` | `eval_params = self._get_parameters(model); self.evaluate(eval_params, config)` inside `fit`. This *also* re-loads the val_loader and re-builds the model inside `evaluate`. | Either compute eval inline (using the already-instantiated model) or remove and rely on Flower's `configure_evaluate` round. |
| C6 | `client_app.py:30-31` | `self.partition_id = int(self.node_config.get("partition-id", 0))`. In current Flower, `partition-id` is an **int already**, but defaulting to `0` masks misconfiguration. | Hard-fail if absent: `node_config["partition-id"]`. |
| C7 | `client_app.py:248-251` | `except Exception as e: ... raise e` re-raises in a way that flattens the traceback. | Use bare `raise` (no variable) to keep the original chain. Ray-backed simulation will then surface a usable trace. |
| C8 | `client_app.py:172` | `train_loss = float(train(model, ...))` returns a per-batch mean — fine — but `num_examples` returned to server (`metrics["num_examples"]`) is `len(train_loader.dataset)`, while the actual `num_examples` for FedAvg weighting is the **first** return value of `fit()`. They agree today, but document this. | Add a unit test asserting parity. |

### 1.3 Strategy (`ProgressivePrivacyStrategy`)

- **Don't shadow base-class behaviors silently.** The class subclasses `FedAvg`
  but never calls `super().aggregate_fit(...)`. Either subclass `Strategy`
  directly (cleaner) or document the contract. Right now `FedAvg.evaluate(...)`
  (the centralized eval hook) is inherited as a no-op because `evaluate_fn` is
  `None`. You also inherit `configure_evaluate` and `aggregate_evaluate`, which
  *do* run — but their results are thrown away (no
  `evaluate_metrics_aggregation_fn`).
- **`configure_fit` sampling** (`server_app.py:115-123`): this branch passes
  `num_clients=self.fraction_fit` (default 1.0 from FedAvg, but pyproject sets
  0.8 → an int-cast 0). The AdaPriv branch (lines 90-113) does the conversion
  correctly. Replace both branches with the standard idiom:
  ```python
  sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
  clients = client_manager.sample(num_clients=sample_size,
                                  min_num_clients=min_num_clients)
  ```
- **Resource-aware sampling skews the privacy amplification.** Subsampling DP
  bounds assume *uniform* sampling of clients. The 70/30 high-resource/random
  split makes `q` non-uniform per client. State this caveat or apply per-client
  amplification factors.
- **`current_parameters` round-trip** (`server_app.py:86`, used at 210). Storing
  the inbound `Parameters` object on the strategy then reconstructing
  `current + delta` is correct because clients return *deltas*, not weights —
  but make sure the docstring on `aggregate_fit` repeats this contract (it
  already says it; promote the warning to the class docstring).
- **`self._client_manager` cache** (`server_app.py:57, 87`): used to compute
  `total_available_clients` for sampling rate. Replace by `client_manager`
  passed into `aggregate_fit` via the standard
  `configure_evaluate`/`aggregate_evaluate` flow — currently you don't have it
  in `aggregate_fit` (it's only in `configure_fit`), hence the cache. Document
  that this is a thread-safety hazard if Flower ever parallelizes strategy
  callbacks.
- **`fit_metrics_aggregation_fn` / `evaluate_metrics_aggregation_fn` not
  passed.** Add:
  ```python
  def weighted_avg(metrics):
      total = sum(n for n, _ in metrics)
      return {k: sum(n * m[k] for n, m in metrics if k in m) / max(total, 1)
              for k in metrics[0][1].keys() if isinstance(metrics[0][1][k], (int, float))}
  ```
  and pass it via `**kwargs` to `super().__init__`. Then the standard Flower
  `History` object will carry your numbers — useful for reviewer audits.
- **`evaluate_fn` (centralized eval) is missing.** A *real* global test set
  evaluation would be cheaper and statistically cleaner than running the test
  set on every client. Add one (see §3.3).

### 1.4 `task.py` separation of concerns

- `make_model` is a 4-line dispatcher and is fine. Move models into a
  `models/` subpackage (they are nontrivial).
- `load_data` mixes (a) IoT toy synthesis, (b) HF dataset routing, (c)
  Normalize transforms, (d) DataLoader construction. Split into:
  - `data/iot.py` — `load_iot(...)`
  - `data/vision.py` — `load_vision(name, partition_id, num_partitions)`
  - `data/transforms.py` — `cifar10_transform()`, `femnist_transform()`
- `_fds_cache` is a process-global. In Ray-backed simulation each actor has its
  own globals — the cache **does not save downloads across SuperNodes**, only
  across calls within one actor. Note this and consider materializing the
  partitioner in a Ray actor's class-level cache or pre-baking partitions.

### 1.5 Simulation / `pyproject.toml`

Current:
```toml
[tool.flwr.federations.local-simulation]
options.num-supernodes = 10
```

This omits `options.backend` entirely. Recommended:
```toml
[tool.flwr.federations.local-simulation]
options.num-supernodes = 10
options.backend.client-resources.num-cpus = 1
options.backend.client-resources.num-gpus = 0.0   # CPU-only sim; bump to 0.1 for 10 clients on 1 GPU
options.backend.init-args.num-cpus = 8            # match host
options.backend.init-args.local-mode = false      # set true to debug serially
options.backend.init-args.runtime-env = { working_dir = "." }
```

Also add a separate federation for CPU-only runs:
```toml
[tool.flwr.federations.local-simulation-cpu]
options.num-supernodes = 10
options.backend.client-resources.num-cpus = 1
options.backend.client-resources.num-gpus = 0.0
```

---

## 2. Scientific & Algorithmic Validity (FL angle)

### 2.1 Data partitioning

- **`IidPartitioner` is the wrong baseline for an IoT FL paper.** `flwr-datasets`
  ships `DirichletPartitioner`, `PathologicalPartitioner`,
  `ShardPartitioner`, `NaturalIdPartitioner`. Recommended:
  ```python
  from flwr_datasets.partitioner import DirichletPartitioner
  partitioner = DirichletPartitioner(num_partitions=N, partition_by="label",
                                     alpha=0.3, seed=seed)
  ```
- Run **at least three** α settings (e.g. 0.1, 0.3, 1.0) to show the algorithm
  works across heterogeneity levels.
- For FEMNIST, use `NaturalIdPartitioner(partition_by="writer_id")` — that is
  the ground-truth FL partitioning and is what reviewers will expect.
- **Validation set leakage check:** `task.py:98` loads the *full test set* on
  every client. There is no per-partition validation split; convergence
  metrics are computed on the same test set the paper reports as held-out. If
  the test set is also used for hyperparameter selection (LR schedule, σ
  bounds), that is **methodological leakage**. Carve a separate
  `validation` split inside the `train` partition.

### 2.2 Aggregation strategy

- **Custom delta-based FedAvg with resource-weighting** (`server_app.py:240-258`).
  Looks correct mathematically:
  `w_i = num_examples_i · resource_score_i`, normalized by `Σ w_i`.
  But the *baseline* (no-DP, fixed-DP) **also** uses `num_examples_i`
  weighting only — fine. Make the choice explicit in the paper: "AdaPriv uses
  resource-weighted FedAvg; baselines use sample-count FedAvg."
- **Should you also report a `FedProx` baseline?** With heterogeneous IoT
  devices, FedAvg drift is a known issue. A FedProx (μ ∈ {0.001, 0.01, 0.1})
  or `FedAvgM` line is a nearly free additional baseline (Flower ships both).
- **`fit_metrics_aggregation_fn`** — see §1.3.
- **Failures handling** (`server_app.py:184-186`): if `results` is empty you
  return `(None, {})`. Flower interprets `None` as "skip this round", which
  is right. But the experiments JSON does *not* record the skipped round —
  add an entry with `skipped: true` so the row count matches `num_rounds`.

### 2.3 Evaluation protocol

Right now you have:
- ✅ Federated training: clients send deltas to server.
- ⚠️  "Federated evaluation": clients each evaluate on the same global test set
  (`task.py:98` returns the full HF test split for every partition_id) →
  this is essentially `n_clients` redundant copies of the same global eval.
- ❌ True centralized evaluation: missing.

**Recommended cleanup:**

1. Make `load_data(...)` return only train data. Move val to a server-side
   helper `load_global_testset(name)`.
2. Pass `evaluate_fn=server_evaluate` to `FedAvg`:
   ```python
   def server_evaluate(server_round, parameters, config):
       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       model = make_model(dataset).to(device)
       set_model_params(model, parameters)
       loss, acc = test(model, GLOBAL_VAL_LOADER, device)
       return loss, {"accuracy": acc, "round": server_round}
   ```
3. Keep client-side evaluation only if you specifically want **per-client**
   metrics (e.g., to show fairness across devices); in that case, give each
   client a true partition-local validation split.

### 2.4 Privacy + FL interaction (Flower-specific)

- The **subsampling rate `q` for the RDP accountant** is computed as
  `len(results) / total_available_clients` (`server_app.py:263`). After the
  resource-aware skew, this is *not* a uniform Bernoulli sampling rate, so the
  RDP amplification bound does not apply. Either: (a) revert to uniform
  sampling for DP runs, or (b) document and use the worst-case `q = 1.0`.
- **Use Opacus or `prv_accountant`** to validate the in-house RDP numbers — see
  `CODE_REVIEW.md` §2 (S4) for details.

---

## 3. Performance & Scaling

### 3.1 Memory leaks in client-side training

| # | File:Line | Problem | Fix |
|---|-----------|---------|-----|
| P1 | `client_app.py:82, 259` | `model = make_model(ds_name).to(device)` every round; old model retained until GC because of `initial_weights_list` references. | After returning, explicitly `del model, train_loader, initial_weights_list; torch.cuda.empty_cache(); gc.collect()`. |
| P2 | `task.py:112-116` | `DataLoader(..., num_workers=2, pin_memory=True)` per call. With Ray simulation and 10 SuperNodes, that is 20 worker processes per round + pinned memory. | Use `num_workers=0` in simulation; pinned memory only when CUDA. |
| P3 | `client_app.py:171` | `train(...)` keeps every layer's gradient until DP application. For FEMNIST + adaptive DP, this is the per-layer storage of `initial_weights_list` — all on GPU. | Move `initial_weights_list` to CPU (`p.detach().clone().cpu()`). |
| P4 | `client_app.py:193-199` | Builds `updated_params` and `initial_weights_numpy` separately, then a third list `updated_deltas`. Three full copies of the model in RAM. | Compute deltas in-place in one pass. |
| P5 | `server_app.py:286-292` | `torch.from_numpy(averaged_delta[i])` to compute one scalar `torch.norm`. Allocates a torch tensor per layer per round. | `np.linalg.norm(averaged_delta[i])` is identical and zero-copy. |
| P6 | `task.py:121` | Optimizer (`SGD`) created fresh each `fit`. Fine, but **gradient memory inside the optimizer is leaked between fits if the optimizer holds references to old buffers** (it doesn't here because it's a local). Document. | — |
| P7 | `client_app.py:135` | The `model.named_parameters()` iteration runs four times in `fit` (lines 86, 122, 161, 179, 192). | Hoist `named_param_list = list(model.named_parameters())` once. |

### 3.2 `fraction_fit` / `min_available_clients`

Current pyproject:
```
num-supernodes = 10
fraction-fit = 0.8       → expected: 8 clients/round
fraction-evaluate = 0.5  → expected: 5 clients/round
min-fit-clients = 3
min-available-clients = 3
```

Bug already noted: `client_manager.sample(num_clients=0.8, min_num_clients=3)`
defaults to `min_num_clients=3` due to int-cast. **Effective fit fraction is
0.3, not 0.8.** Verify this empirically:

```python
# in aggregate_fit
print(f"DEBUG: num_results={len(results)} of {self._client_manager.num_available()}")
```

After the fix (use `num_fit_clients`), confirm 8 clients participate per round.

### 3.3 Centralized vs federated evaluation cost

With 10 clients × 50 rounds × the full CIFAR-10 test set (10k images), client
eval performs 5,000,000 forward passes per run. A single centralized
`evaluate_fn` would do 500,000 — a **10× speedup** on evaluation alone.

---

## 4. Testing & Results Extraction

### 4.1 Pytest suite — Flower-specific tests

Create `tests/flower/`:

```
tests/flower/
├── conftest.py
├── test_client_state_dict.py     # parameter round-trip via state_dict
├── test_strategy_unit.py         # strategy methods with mocked ClientProxies
├── test_partitioner.py           # IID vs Dirichlet split sanity checks
├── test_simulation_smoke.py      # 1-round end-to-end via flwr run
└── test_metrics_aggregation.py   # weighted_avg correctness
```

#### `tests/flower/test_client_state_dict.py`

```python
import numpy as np, torch
from qpriviot_fl.task import make_model
from qpriviot_fl.client_app import QPrivIoTClient

class _Ctx:
    run_config = {"dataset": "cifar10", "local-epochs": 1, "learning-rate": 0.01}
    node_config = {"partition-id": 0, "num-partitions": 1}

def test_parameter_roundtrip_preserves_buffers(monkeypatch):
    model_a = make_model("cifar10")
    arrays = [p.detach().cpu().numpy() for _, p in model_a.named_parameters()]
    model_b = make_model("cifar10")
    client = QPrivIoTClient.__new__(QPrivIoTClient)  # bypass __init__
    client._set_parameters(model_b, arrays, torch.device("cpu"))
    for (n, a), (_, b) in zip(model_a.named_parameters(), model_b.named_parameters()):
        np.testing.assert_allclose(a.detach().numpy(), b.detach().numpy())
```

#### `tests/flower/test_strategy_unit.py`

```python
import numpy as np, pytest
from unittest.mock import MagicMock
from flwr.common import ndarrays_to_parameters, FitRes, Status, Code
from qpriviot_fl.server_app import ProgressivePrivacyStrategy
from qpriviot_fl.task import make_model

def _mock_fitres(deltas, n=100, **metrics):
    return FitRes(status=Status(Code.OK, ""), parameters=ndarrays_to_parameters(deltas),
                  num_examples=n, metrics={"train_loss": 0.5, "resource_score": 0.8,
                                           "avg_noise": 1.0, "dp_mode": "Fixed",
                                           **metrics})

def test_aggregate_returns_none_on_empty():
    s = ProgressivePrivacyStrategy(num_rounds=1, dataset="cifar10")
    assert s.aggregate_fit(1, [], []) == (None, {})

def test_weighted_avg_two_clients():
    s = ProgressivePrivacyStrategy(num_rounds=1, dataset="cifar10",
                                   use_adaptive_dp=False)
    init_params = [p.detach().cpu().numpy() for _, p in make_model("cifar10").named_parameters()]
    s.current_parameters = ndarrays_to_parameters(init_params)
    d1 = [np.ones_like(p) for p in init_params]
    d2 = [np.full_like(p, 3.0) for p in init_params]
    p1, p2 = MagicMock(cid="c1"), MagicMock(cid="c2")
    res1, res2 = _mock_fitres(d1, n=100), _mock_fitres(d2, n=300)
    out, _ = s.aggregate_fit(1, [(p1, res1), (p2, res2)], [])
    # Weighted by num_examples: (1*100 + 3*300) / 400 = 2.5
    out_arrays = [a for a in __import__("flwr.common").common.parameters_to_ndarrays(out)]
    np.testing.assert_allclose(out_arrays[0], init_params[0] + 2.5)
```

#### `tests/flower/test_simulation_smoke.py`

```python
import json, os, pathlib, subprocess, sys
import pytest

@pytest.mark.slow
def test_one_round_iot_simulation(tmp_path, monkeypatch):
    repo = pathlib.Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    cmd = [sys.executable, "-m", "flwr", "run", str(repo), "local-simulation",
           "--run-config",
           'dataset="iot" num-server-rounds=1 use-dp=false use-adaptive-dp=false']
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    out = list(tmp_path.glob("results_*.json"))
    assert out, "results_*.json not produced"
    payload = json.loads(out[0].read_text())
    assert len(payload["rounds"]) == 1
    assert "val_accuracy" in payload["rounds"][0]
```

#### `tests/flower/test_partitioner.py`

```python
import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner

def test_dirichlet_creates_imbalance():
    p = DirichletPartitioner(num_partitions=10, partition_by="label", alpha=0.3, seed=1)
    fds = FederatedDataset(dataset="uoft-cs/cifar10", partitioners={"train": p})
    sizes = [len(fds.load_partition(i, "train")) for i in range(10)]
    cv = np.std(sizes) / np.mean(sizes)
    assert cv > 0.10, f"Dirichlet partitions are too uniform (cv={cv:.2f})"
```

### 4.2 Mock client/server FL loop test

Use the in-process simulation API for fast unit-level checks:

```python
from flwr.simulation import run_simulation
from flwr.client import ClientApp, NumPyClient
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Context
import numpy as np, torch, torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self): super().__init__(); self.l = nn.Linear(2, 2)
    def forward(self, x): return self.l(x)

class TinyClient(NumPyClient):
    def __init__(self, ctx):
        self.m = TinyNet()
    def get_parameters(self, _):
        return [v.detach().numpy() for v in self.m.state_dict().values()]
    def fit(self, params, _):
        for k, v in zip(self.m.state_dict().keys(), params):
            self.m.state_dict()[k].copy_(torch.from_numpy(v))
        return self.get_parameters({}), 1, {"loss": 0.0}
    def evaluate(self, params, _):
        return 0.0, 1, {"accuracy": 1.0}

def client_fn(ctx: Context): return TinyClient(ctx).to_client()
def server_fn(ctx: Context):
    return ServerAppComponents(strategy=FedAvg(min_fit_clients=2, min_available_clients=2),
                               config=ServerConfig(num_rounds=2))

def test_smoke_simulation():
    run_simulation(server_app=ServerApp(server_fn=server_fn),
                   client_app=ClientApp(client_fn=client_fn),
                   num_supernodes=2)
```

This test runs in <5 s and verifies the end-to-end Flower wiring without
hitting any HF dataset.

### 4.3 Structured logging (paper-grade)

Add a `qpriviot_fl/logging_utils.py`:

```python
import json, csv, os, time
from typing import Dict

class RunLogger:
    """Mirrors metrics to JSON, CSV, and (optionally) W&B."""
    def __init__(self, out_dir: str, run_name: str, use_wandb: bool = False):
        os.makedirs(out_dir, exist_ok=True)
        self.json_path = os.path.join(out_dir, f"{run_name}.json")
        self.csv_path = os.path.join(out_dir, f"{run_name}.csv")
        self.rows = []
        self.t0 = time.time()
        self.wandb = None
        if use_wandb:
            import wandb
            self.wandb = wandb.init(project="qpriviot-fl", name=run_name,
                                    config={}, reinit=True)

    def log(self, row: Dict):
        row["wall_time_s"] = round(time.time() - self.t0, 3)
        self.rows.append(row)
        if self.wandb: self.wandb.log(row)

    def save(self):
        with open(self.json_path, "w") as f:
            json.dump({"rounds": self.rows}, f, indent=2)
        if self.rows:
            with open(self.csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=sorted({k for r in self.rows for k in r}))
                w.writeheader(); w.writerows(self.rows)
        if self.wandb: self.wandb.finish()
```

Replace `_save` in `ProgressivePrivacyStrategy` with calls into `RunLogger`.
Add `use-wandb=false` to `[tool.flwr.app.config]`. The W&B integration is
gated behind that flag so the artifact still works offline.

---

## 5. Stylistic Suggestions (Flower-idiomatic)

- Use Flower's `INFO`/`WARNING` log calls instead of `print`:
  ```python
  from flwr.common.logger import log
  from logging import INFO, WARNING
  log(INFO, "Round %s | DP=%s | ε=%.3f", server_round, dp_mode, epsilon_t)
  ```
  Logs are then routed through Flower's `History` / Ray logging consistently.
- Replace literal config strings (`"use_secagg"`, `"epsilon_t"`, etc.) with a
  module-level `CONFIG_KEYS` `Enum` or constants — typo bugs in config
  dictionaries are a common source of silent failures.
- Type the strategy public methods with `flwr.common` types (`FitIns`,
  `FitRes`, `Parameters`, `Scalar`) — your IDE will surface the
  `client_manager.sample` signature error.
- Drop the `from __future__ import annotations` *and* `Optional` mix in
  `server_app.py:1, 9` — pick one (3.10+: `X | None`).
- Move `DEVICE_MAP` (`client_app.py:21`) into `device_profile.py` so the
  enumeration is single-source-of-truth.

---

## 6. Testing & Execution Guide (terminal)

### 6.1 Install

```bash
pip install -e '.[dev]'
# add: pytest pytest-cov pytest-xdist hypothesis flwr-datasets[vision] opacus
```

### 6.2 Lint / type / unit tests

```bash
ruff check qpriviot_fl tests
black --check -l 100 qpriviot_fl tests
mypy --ignore-missing-imports qpriviot_fl

mkdir -p test_artifacts
pytest tests/unit tests/flower -v -m "not slow" \
  --cov=qpriviot_fl --cov-report=xml:test_artifacts/coverage.xml \
  --junitxml=test_artifacts/junit_unit.xml \
  | tee test_artifacts/unit.log
```

### 6.3 Smoke simulation (1 round, IoT toy)

```bash
flwr run . local-simulation \
  --run-config 'dataset="iot" num-server-rounds=1 use-dp=false use-adaptive-dp=false'
```

### 6.4 Reproducible main comparison (50 rounds)

After fixing the sampling bug and adding a `--seed` plumb-through:

```bash
SEED=1337
for cfg in "use-dp=false use-adaptive-dp=false" \
           "use-dp=true use-adaptive-dp=false" \
           "use-dp=true use-adaptive-dp=true"; do
  name=$(echo "$cfg" | tr ' =' '__')
  flwr run . local-simulation \
    --run-config "dataset=\"cifar10\" num-server-rounds=50 target-epsilon=3.0 \
                  learning-rate=0.001 seed=$SEED $cfg" \
    2>&1 | tee logs/${name}_seed${SEED}.log
done
```

### 6.5 Privacy-utility tradeoff sweep

```bash
for eps in 0.5 1 2 3 5 8 10; do
  for cfg in "use-dp=true use-adaptive-dp=false" "use-dp=true use-adaptive-dp=true"; do
    flwr run . local-simulation \
      --run-config "dataset=\"cifar10\" num-server-rounds=50 \
                    target-epsilon=${eps} seed=1337 ${cfg}" \
      2>&1 | tee logs/eps${eps}_${cfg// /_}.log
  done
done
```

### 6.6 GPU simulation (1 GPU, 10 SuperNodes)

After patching `pyproject.toml`:
```toml
options.backend.client-resources.num-gpus = 0.1
```
Then:
```bash
CUDA_VISIBLE_DEVICES=0 flwr run . local-simulation --run-config 'dataset="cifar10" num-server-rounds=50'
```

### 6.7 Plot generation

```bash
python plot.py    # reads ./experiment_results/, writes ./figures/
```

### 6.8 Result aggregation for the appendix

```bash
python -c "
import glob, json, csv
rows = []
for f in glob.glob('experiment_results/*.json'):
    d = json.load(open(f))
    for r in d['rounds']:
        r['file'] = f; rows.append(r)
keys = sorted({k for r in rows for k in r})
with open('test_artifacts/all_results.csv','w',newline='') as out:
    w = csv.DictWriter(out, fieldnames=keys); w.writeheader(); w.writerows(rows)
print(f'Wrote {len(rows)} rows × {len(keys)} cols')
"
```

---

## 7. Priority-Ordered Fix List

**Must-fix (blocks reproducibility/validity)**
1. `ClientManager.sample` int-vs-fraction bug (`server_app.py:115-123`).
2. Add `Dirichlet` (or `NaturalId`) partitioner; document α settings.
3. Add `evaluate_fn` (centralized eval) + drop redundant client-side eval-in-fit.
4. Switch parameter sync to `state_dict()` round-trip.
5. Configure `options.backend.client-resources` in `pyproject.toml`.
6. Make `secagg_seed` derivable from a config seed.

**Should-fix**
7. `fit_metrics_aggregation_fn` + `evaluate_metrics_aggregation_fn`.
8. Add `torch.cuda.empty_cache()` + `del model` after each `fit`/`evaluate`.
9. Cache `(train_loader, val_loader)` per client instance.
10. Replace `print` with `flwr.common.logger.log`.
11. Add the `tests/flower/` suite (smoke + strategy unit + state-dict).

**Nice-to-have**
12. FedProx / FedAvgM baseline lines for the paper.
13. W&B integration via `RunLogger`.
14. Structured run-config keys via `Enum`.