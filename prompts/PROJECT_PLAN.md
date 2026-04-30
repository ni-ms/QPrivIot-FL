# QPrivIoT-FL — Project Review & Execution Plan

A reviewer-grade reference for setting up, testing, executing, and monitoring
the Flower federated-learning project in this repository. Each section is
self-contained: a project lead can hand it to a new collaborator.

---

## 1. Project Structure Review

### 1.1 Required Flower app layout

A canonical Flower app contains the following files. Tick each as you verify it
in the working tree:

- [ ] `pyproject.toml` at repo root — declares `[project]`, `[build-system]`,
      `[tool.flwr.app.components]` and `[tool.flwr.federations.<name>]`.
- [ ] `qpriviot_fl/__init__.py` — re-exports the public API.
- [ ] `qpriviot_fl/server_app.py` — defines `app = ServerApp(server_fn=...)`.
- [ ] `qpriviot_fl/client_app.py` — defines `app = ClientApp(client_fn=...)`.
- [ ] `qpriviot_fl/task.py` — model + dataloader factories (`make_model`,
      `load_data`, `train`, `test`, `get_weights`, `set_weights`).
- [ ] `qpriviot_fl/config.py` — typed dataclasses for non-run-config defaults.
- [ ] `tests/` — pytest suite with `conftest.py` for fixtures.
- [ ] `experiment_results/` — JSON outputs (gitignored).
- [ ] `figures/` — generated plots (gitignored).
- [ ] `README.md` — install + run instructions.

### 1.2 `pyproject.toml` audit checklist

- [ ] `[tool.flwr.app.components].serverapp = "qpriviot_fl.server_app:app"`
- [ ] `[tool.flwr.app.components].clientapp = "qpriviot_fl.client_app:app"`
- [ ] All run-tunable parameters live under `[tool.flwr.app.config]`
      (so they are overridable with `--run-config`).
- [ ] At least one federation in `[tool.flwr.federations.<name>]` with
      `options.num-supernodes` and `options.backend.client-resources`.
- [ ] `[project].dependencies` pins `flwr[simulation]`, `flwr-datasets`,
      `torch`, `torchvision`, `numpy`, `matplotlib`, plus any privacy libs.
- [ ] `[project.optional-dependencies].dev` lists `pytest`, `pytest-cov`,
      `pytest-xdist`, `ruff`, `black`, `opacus`.

### 1.3 Code-organization checklist (Flower-specific)

- [ ] `ClientApp` is constructed with `client_fn=client_fn` returning
      `NumPyClient(...).to_client()`.
- [ ] `ServerApp` is constructed with `server_fn=server_fn` returning
      `ServerAppComponents(strategy=..., config=ServerConfig(...))`.
- [ ] Parameter sync uses `state_dict()` round-trip (`get_weights` /
      `set_weights`) so buffers are propagated.
- [ ] Strategy passes `fit_metrics_aggregation_fn` and
      `evaluate_metrics_aggregation_fn` so Flower's `History` carries metrics.
- [ ] Centralized eval is set via `evaluate_fn=...` rather than reproduced
      inside each client's `fit()`.
- [ ] No module-level RNG state leaks between SuperNodes (use seeded local
      `random.Random()` / `np.random.Generator` instances).

---

## 2. Environment Setup

### 2.1 One-time setup

```bash
# 1. Clone and enter the repo (if not already there)
git clone <your-fork-url> QPrivIot-FL
cd QPrivIot-FL

# 2. Create an isolated virtual environment (Python ≥ 3.10)
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Upgrade build tooling
pip install --upgrade pip setuptools wheel

# 4. Install the project + dev extras in editable mode
pip install -e '.[dev]'

# 5. Verify the install
python -c "import qpriviot_fl; print('qpriviot_fl', qpriviot_fl.__version__)"
flwr --version
```

If `pip install -e '.[dev]'` fails because the `dev` extra is not yet declared,
add this block to `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
  "pytest>=8",
  "pytest-cov",
  "pytest-xdist",
  "ruff",
  "black",
  "opacus",
  "wandb",
]
```

### 2.2 Federation configuration

Open `pyproject.toml` and confirm a simulation federation exists. The
recommended block for a 10-client local simulation:

```toml
[tool.flwr.federations]
default = "local-simulation"

[tool.flwr.federations.local-simulation]
options.num-supernodes = 10
options.backend.client-resources.num-cpus = 2
options.backend.client-resources.num-gpus = 0.0          # bump to 0.1 for GPU sim

[tool.flwr.federations.local-simulation-gpu]
options.num-supernodes = 10
options.backend.client-resources.num-cpus = 1
options.backend.client-resources.num-gpus = 0.1          # 10 supernodes × 0.1 = 1 GPU

[tool.flwr.federations.remote-federation]
address = "<SUPERLINK-ADDRESS>:<PORT>"
insecure = true
```

Run-config defaults (override at the CLI with `--run-config`):

```toml
[tool.flwr.app.config]
num-server-rounds = 50
fraction-fit = 0.8
fraction-evaluate = 0.5
min-fit-clients = 3
min-available-clients = 3
batch-size = 32
local-epochs = 5
learning-rate = 0.05
dataset = "cifar10"
seed = 1337
dirichlet-alpha = 0.3
use-secagg = false
use-dp = false
use-adaptive-dp = false
target-epsilon = 12.0
base-noise = 0.3
```

### 2.3 Reproducibility hygiene

- [ ] `seed` is wired through `server_fn` to `torch.manual_seed`,
      `np.random.seed`, `random.seed`, and DataLoader `worker_init_fn`.
- [ ] Record git SHA in every results file:
  ```python
  import subprocess
  sha = subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()
  ```
- [ ] Capture environment for the appendix:
  ```bash
  python -V > test_artifacts/env.txt
  pip freeze >> test_artifacts/env.txt
  ```

---

## 3. Testing Procedures

### 3.1 Local simulation (smoke run)

The fastest end-to-end check — runs the full server/client loop against the
synthetic IoT dataset for one round:

```bash
flwr run . local-simulation \
  --run-config 'dataset="iot" num-server-rounds=1 use-dp=false use-adaptive-dp=false'
```

Expected output: a `results_no_dp.json` file with one round entry, and a
`global_model.npz` checkpoint.

### 3.2 Unit & integration tests

The project ships a `tests/` directory with `conftest.py` registering the
`slow` marker and an autouse deterministic seed fixture.

```bash
# Fast suite — unit tests only
pytest tests/ -v -m "not slow"

# Full suite, with coverage and JUnit output for the appendix
mkdir -p test_artifacts
pytest tests/ -v \
  --cov=qpriviot_fl \
  --cov-report=term-missing \
  --cov-report=xml:test_artifacts/coverage.xml \
  --junitxml=test_artifacts/junit.xml \
  | tee test_artifacts/pytest.log

# Parallelize on multi-core machines
pytest tests/ -n auto
```

### 3.3 Unit testing the `ClientApp`

A `NumPyClient` is testable directly without spinning up Flower's runtime:

```python
# tests/flower/test_client_unit.py
import numpy as np, torch
from qpriviot_fl.task import make_model, get_weights, set_weights
from qpriviot_fl.client_app import QPrivIoTClient

def test_set_and_get_weights_roundtrip():
    model = make_model("cifar10")
    expected = get_weights(model)
    fresh = make_model("cifar10")
    client = QPrivIoTClient.__new__(QPrivIoTClient)   # bypass __init__
    client._set_parameters(fresh, expected, torch.device("cpu"))
    actual = get_weights(fresh)
    for a, b in zip(expected, actual):
        np.testing.assert_allclose(a, b, atol=1e-6)
```

### 3.4 Unit testing the `ServerApp` strategy

Mock `ClientProxy` / `FitRes` to feed fake client responses into
`aggregate_fit`:

```python
# tests/flower/test_server_unit.py
import numpy as np
from unittest.mock import MagicMock
from flwr.common import ndarrays_to_parameters, FitRes, Status, Code, parameters_to_ndarrays
from qpriviot_fl.server_app import ProgressivePrivacyStrategy
from qpriviot_fl.task import make_model, get_weights

def _fitres(deltas, n=100):
    return FitRes(status=Status(Code.OK, ""),
                  parameters=ndarrays_to_parameters(deltas),
                  num_examples=n,
                  metrics={"train_loss": 0.5, "resource_score": 0.8,
                           "avg_noise": 0.0, "dp_mode": "None"})

def test_weighted_fedavg_two_clients():
    s = ProgressivePrivacyStrategy(num_rounds=1, dataset="cifar10",
                                   use_adaptive_dp=False, target_epsilon=100.0)
    init = get_weights(make_model("cifar10"))
    s.current_parameters = ndarrays_to_parameters(init)
    d1 = [np.ones_like(p) for p in init]
    d2 = [np.full_like(p, 3.0) for p in init]
    p1, p2 = MagicMock(cid="c1"), MagicMock(cid="c2")
    out, _ = s.aggregate_fit(1, [(p1, _fitres(d1, 100)),
                                 (p2, _fitres(d2, 300))], [])
    arrs = list(parameters_to_ndarrays(out))
    np.testing.assert_allclose(arrs[0], init[0] + 2.5)   # weighted mean
```

### 3.5 In-process simulation test

For a fast end-to-end test that does not shell out to `flwr run`:

```python
# tests/flower/test_inprocess_sim.py
from flwr.simulation import run_simulation
from qpriviot_fl.client_app import app as client_app
from qpriviot_fl.server_app import app as server_app

def test_one_round_inprocess():
    run_simulation(server_app=server_app, client_app=client_app,
                   num_supernodes=2,
                   backend_config={"client_resources": {"num_cpus": 1}})
```

### 3.6 Linting / formatting gate

```bash
ruff check qpriviot_fl tests
black --check -l 100 qpriviot_fl tests
```

---

## 4. Experiment Execution

All experiments are launched through `flwr run`. Override any value declared
under `[tool.flwr.app.config]` via `--run-config "key=value [key2=value2 ...]"`.

### 4.1 Three-mode main comparison (50 rounds)

```bash
mkdir -p logs

# 1) No DP baseline
flwr run . local-simulation \
  --run-config 'dataset="cifar10" num-server-rounds=50 learning-rate=0.001
                use-dp=false use-adaptive-dp=false seed=1337' \
  2>&1 | tee logs/main_no_dp_seed1337.log
mv results_no_dp.json experiment_results/results_cifar10_no-dp_seed1337_eps3.0.json

# 2) Fixed DP at ε=3.0
flwr run . local-simulation \
  --run-config 'dataset="cifar10" num-server-rounds=50 learning-rate=0.001
                use-dp=true use-adaptive-dp=false target-epsilon=3.0 seed=1337' \
  2>&1 | tee logs/main_fixed_dp_seed1337.log
mv results_uniform_dp.json experiment_results/results_cifar10_fixed-dp_seed1337_eps3.0.json

# 3) AdaPriv (ours) at ε=3.0
flwr run . local-simulation \
  --run-config 'dataset="cifar10" num-server-rounds=50 learning-rate=0.001
                use-dp=true use-adaptive-dp=true target-epsilon=3.0 seed=1337' \
  2>&1 | tee logs/main_adapriv_seed1337.log
mv results_adaptive_dp.json experiment_results/results_cifar10_adapriv_seed1337_eps3.0.json
```

### 4.2 Privacy-utility trade-off sweep

```bash
for eps in 0.5 1.0 2.0 3.0 5.0 8.0 10.0; do
  for cfg in 'use-dp=true use-adaptive-dp=false' \
             'use-dp=true use-adaptive-dp=true'; do
    flwr run . local-simulation \
      --run-config "dataset=\"cifar10\" num-server-rounds=50 \
                    learning-rate=0.001 target-epsilon=${eps} seed=1337 ${cfg}"
  done
done
```

### 4.3 Multi-seed robustness (3-seed bar plots)

```bash
for seed in 1337 2024 9001; do
  for cfg in 'use-dp=false use-adaptive-dp=false' \
             'use-dp=true use-adaptive-dp=false' \
             'use-dp=true use-adaptive-dp=true'; do
    flwr run . local-simulation \
      --run-config "dataset=\"cifar10\" num-server-rounds=50 \
                    learning-rate=0.001 target-epsilon=3.0 seed=${seed} ${cfg}"
  done
done
```

### 4.4 Non-IID sensitivity sweep

```bash
for alpha in 0.1 0.3 1.0; do
  flwr run . local-simulation \
    --run-config "dataset=\"cifar10\" dirichlet-alpha=${alpha} \
                  num-server-rounds=50 use-dp=true use-adaptive-dp=true \
                  target-epsilon=3.0 seed=1337"
done
```

### 4.5 Hyperparameter override examples

```bash
# Larger fit fraction, smaller LR
flwr run . local-simulation \
  --run-config 'fraction-fit=1.0 learning-rate=0.005'

# Heavier local computation per round
flwr run . local-simulation \
  --run-config 'local-epochs=10 batch-size=64'

# SecAgg + DP combination
flwr run . local-simulation \
  --run-config 'use-secagg=true use-dp=true target-epsilon=5.0'

# Switch dataset
flwr run . local-simulation \
  --run-config 'dataset="femnist" num-server-rounds=20'
```

### 4.6 GPU simulation

```bash
CUDA_VISIBLE_DEVICES=0 flwr run . local-simulation-gpu \
  --run-config 'dataset="cifar10" num-server-rounds=50 use-adaptive-dp=true'
```

### 4.7 Plot generation

```bash
python plot.py
ls figures/    # graph1_*.png ... graph7_*.png
```

---

## 5. Monitoring & Metrics

### 5.1 Terminal-log monitoring

Every round prints a one-line summary:

```
--- Round 12 [DP: Adaptive, SecAgg: Off, Conv: 0.78, ε_t: 0.412] ---
  Aggregating Deltas (Resource-Weighted FedAvg)...
  Total Epsilon: 1.84 | Avg. Loss: 0.6213 | Val Acc: 0.7124
```

Tail a long run in a second terminal:

```bash
tail -f logs/main_adapriv_seed1337.log
```

Filter only round summaries:

```bash
grep -E 'Round|Total Epsilon' logs/main_adapriv_seed1337.log
```

### 5.2 Structured JSON logs

Each strategy writes to `results_<mode>.json` with one entry per round. Schema:

```json
{
  "rounds": [
    {
      "round": 1,
      "avg_loss": 1.823,
      "val_accuracy": 0.412,
      "val_loss": 1.612,
      "convergence_score": 0.0,
      "total_epsilon": 0.21,
      "epsilon_t": 0.45,
      "sensitivities": {"conv1.weight": 1.04, "fc2.bias": 0.91},
      "avg_latency": 0.18,
      "avg_resource": 0.71,
      "clip_norm": 4.83,
      "avg_sensitivity": 1.0,
      "secagg_active": false,
      "dp_mode": "Adaptive"
    }
  ]
}
```

### 5.3 Aggregating runs into a single CSV

```bash
python - <<'PY'
import glob, json, csv
rows = []
for f in glob.glob("experiment_results/*.json"):
    payload = json.load(open(f))
    for r in payload["rounds"]:
        r["file"] = f
        rows.append(r)
keys = sorted({k for r in rows for k in r})
with open("test_artifacts/all_results.csv", "w", newline="") as out:
    w = csv.DictWriter(out, fieldnames=keys)
    w.writeheader(); w.writerows(rows)
print(f"Wrote {len(rows)} rows × {len(keys)} cols")
PY
```

### 5.4 Weights & Biases integration

Add a thin run-logger so W&B is opt-in via `--run-config use-wandb=true`:

```python
# qpriviot_fl/wandb_logger.py
import os
import wandb

def init_wandb(run_config: dict, run_name: str):
    if not run_config.get("use-wandb", False):
        return None
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "qpriviot-fl"),
        name=run_name,
        config=dict(run_config),
        reinit=True,
    )

def log_round(run, record: dict):
    if run is None:
        return
    flat = {k: v for k, v in record.items() if isinstance(v, (int, float, bool))}
    run.log(flat, step=record["round"])
```

Wire into the strategy's `aggregate_fit`:

```python
# server_app.py
from qpriviot_fl.wandb_logger import init_wandb, log_round

# in __init__ or server_fn
self._wandb = init_wandb(context.run_config,
                         run_name=os.path.basename(self.results_file))

# at the end of aggregate_fit
log_round(self._wandb, record)
```

Authenticate once, then run with W&B enabled:

```bash
export WANDB_API_KEY=...
export WANDB_PROJECT=qpriviot-fl

flwr run . local-simulation \
  --run-config 'use-wandb=true dataset="cifar10" num-server-rounds=50
                use-adaptive-dp=true target-epsilon=3.0 seed=1337'
```

W&B dashboards to set up:
- **Privacy budget evolution** — line plot of `total_epsilon` vs `round`.
- **Utility curves** — `val_accuracy` and `avg_loss` vs `round`, grouped by
  `dp_mode`.
- **Resource selection fairness** — histogram of `avg_resource` across runs.
- **Per-layer sensitivity heatmap** — table of `sensitivities` over rounds.

### 5.5 What to monitor in real time

- **Loss divergence** — if `avg_loss` increases for ≥3 consecutive rounds,
  abort and lower `learning-rate`.
- **Privacy budget exhaustion** — if `total_epsilon` exceeds `target-epsilon`,
  the strategy halts; check the log for `STOPPING: Privacy budget exceeded`.
- **Client dropouts** — `avg_resource` < `min_eligible_score` means too many
  devices are being skipped; raise `min-available-clients` or revisit the
  device-profile distribution.
- **NaN propagation** — any round where `avg_sensitivity` or `clip_norm` is
  `NaN` indicates an empty-list `np.mean`; guard those calls in the strategy.

---

## 6. Pre-flight Checklist Before Running a Paper Experiment

- [ ] `pip install -e '.[dev]'` completes cleanly in a fresh `.venv`.
- [ ] `pytest tests/ -m "not slow"` passes.
- [ ] `flwr run . local-simulation --run-config 'dataset="iot" num-server-rounds=1'`
      succeeds (smoke test).
- [ ] `pyproject.toml` `[tool.flwr.app.config]` matches the values quoted in
      the paper.
- [ ] `git rev-parse HEAD` is recorded in every `experiment_results/*.json`.
- [ ] Each unique experiment ran with at least 3 seeds.
- [ ] `python plot.py` regenerates all 7 figures without errors.
- [ ] CSV/JSON appendix artifacts are produced under `test_artifacts/`.