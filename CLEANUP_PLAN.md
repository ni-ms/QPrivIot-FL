# QPrivIoT-FL — Repository Cleanup Plan

A step-by-step, low-risk plan to tidy the project directory before submission.
Each step is reversible (everything goes through git or a timestamped archive
folder) and ends with a verification command.

---

## Step 0 — Take a Safety Net

Before *any* deletion, make the cleanup auditable.

```bash
# 1. Confirm the working tree is clean enough to recover from
git status

# 2. Stash anything in-flight you do not want to commit yet
git stash push -u -m "pre-cleanup stash"

# 3. Tag the current HEAD so you can return to it later
git tag pre-cleanup-$(date +%Y%m%d)

# 4. Create an archive folder *outside* the repo for files you remove but
#    are not 100% sure about
mkdir -p ../qpriviot-archive/$(date +%Y%m%d)
```

If anything goes wrong: `git reset --hard pre-cleanup-<date>` restores the tree.

---

## Step 1 — Code Organization

Goal: a clean `qpriviot_fl/` package where `server_app.py`, `client_app.py`,
and `task.py` each have a single responsibility.

### 1.1 Recommended layout

```
qpriviot_fl/
├── __init__.py            # public API re-exports
├── server_app.py          # ServerApp + Strategy ONLY
├── client_app.py          # ClientApp + NumPyClient ONLY
├── task.py                # data loading, training, evaluation
├── config.py              # @dataclass run-config defaults
├── device_profile.py      # device profiling
├── privacy_utils.py       # DP / SecAgg
└── models/                # (optional) one model per file when >1 architecture
    ├── __init__.py
    ├── cifar.py
    ├── femnist.py
    └── iot.py
```

### 1.2 Single-responsibility checklist

For each file, tick once you have refactored anything that does not belong:

- [ ] `server_app.py` — contains only `ServerApp(server_fn=...)`, the
      strategy class, `server_fn(context)`. **No** model definitions, **no**
      data-loading. Imports from `task.py` for `make_model`, `get_weights`.
- [ ] `client_app.py` — contains only `ClientApp(client_fn=...)`, the
      `NumPyClient` subclass, `client_fn(context)`. **No** privacy math
      (lives in `privacy_utils.py`), **no** model classes.
- [ ] `task.py` — `make_model`, `load_data`, `train`, `test`,
      `get_weights`, `set_weights`. **No** Flower imports
      (`flwr.server`, `flwr.client`) here.
- [ ] `config.py` — only `@dataclass` definitions and a `DEFAULT_CONFIG`
      instance. **No** runtime logic.
- [ ] `privacy_utils.py` — DP accountant, sensitivity tracker, SecAgg.
      **No** model imports.
- [ ] `__init__.py` — only re-exports. No side-effect code.

### 1.3 Move plotting and one-off scripts to `scripts/`

Top-level scripts pollute the package import path. Move them:

```bash
mkdir -p scripts
git mv plot.py                       scripts/plot.py
git mv quick_proof.sh                scripts/quick_proof.sh
git mv run_main_comparison.sh        scripts/run_main_comparison.sh
git mv run_mini_suite.sh             scripts/run_mini_suite.sh
git mv run_experiments.sh            scripts/run_experiments.sh
git mv run_comprehensive_suite.sh    scripts/run_comprehensive_suite.sh
```

Update any references inside the scripts (e.g. `python plot.py` →
`python scripts/plot.py`) and any README mentions.

### 1.4 Promote tests to a top-level peer of the package

Already partially done — verify:

```bash
ls tests/                       # conftest.py, __init__.py, flower/
ls qpriviot_fl/test 2>/dev/null # legacy folder?
```

If `qpriviot_fl/test/` exists and contains only docs / non-test modules, move
those out:

```bash
mkdir -p docs/research_notes
git mv qpriviot_fl/test/RESEARCH_SUMMARY.md       docs/research_notes/
git mv qpriviot_fl/test/algorithms_documentation.md docs/research_notes/
git mv qpriviot_fl/test/formal_algorithms.md      docs/research_notes/
git mv qpriviot_fl/test/config.yaml               docs/research_notes/
# metrics.py / privacy.py: decide — port into qpriviot_fl/ or delete
git rm -r qpriviot_fl/test/
```

### 1.5 Verification

```bash
# Imports still work
python -c "from qpriviot_fl import server_app, client_app, task; print('ok')"

# Flower app entry points still resolve
python -c "from qpriviot_fl.server_app import app as s; \
           from qpriviot_fl.client_app import app as c; print(type(s), type(c))"

# Tests still collect
pytest tests/ --collect-only -q
```

---

## Step 2 — Standard Exclusions (`.gitignore`)

The current `.gitignore` covers Python basics and `.flwr`. Append the
**Flower- and ML-specific** patterns below.

### 2.1 Patterns to add

```bash
cat >> .gitignore <<'EOF'

# ───── Flower simulation outputs ─────
.flwr/
.flwr-cache/
flwr.log
flwr-server-*.log
flwr-client-*.log

# ───── Experiment artifacts (regenerable) ─────
results.json
results_*.json
experiment_results/
logs/
test_artifacts/
figures/
*.npz
*.pt
*.pth
*.ckpt

# ───── Hugging Face / Ray / W&B caches ─────
.cache/
~/.cache/huggingface/
.huggingface/
.ray/
ray_results/
wandb/
.wandb/

# ───── Pytest / coverage ─────
.pytest_cache/
.mypy_cache/
.ruff_cache/

# ───── IDE / OS noise ─────
.idea/
.vscode/
.DS_Store
*.swp
*.swo

# ───── Notebook checkpoints ─────
.ipynb_checkpoints/

# ───── Local environments ─────
.venv/
venv/
env/
.env
.env.*
EOF
```

### 2.2 Stop tracking files that are already in git but should be ignored

After updating `.gitignore`, untrack the files Git is currently watching:

```bash
git rm --cached -r .DS_Store .idea/ 2>/dev/null
git rm --cached global_model.npz results.json results_no_dp.json 2>/dev/null
git rm --cached -r experiment_results/ figures/ 2>/dev/null

git status   # verify only the intended deletions are staged
```

> ⚠️ Do **not** commit yet — first verify the next steps.

### 2.3 Whitelisting an example output (optional)

If you want the appendix to ship one canonical results file, force-add it:

```bash
git add -f experiment_results/results_cifar10_adapriv_seed1337_eps3.0.json
```

### 2.4 Verification

```bash
# Anything still tracked that the new .gitignore says shouldn't be?
git ls-files -ci --exclude-standard

# Expect: empty output. If non-empty, untrack each entry above.
```

---

## Step 3 — Manual Verification of `pyproject.toml`

`pyproject.toml` is the single source of truth for the Flower app — a
careless edit can silently break `flwr run`.

### 3.1 Mandatory keys checklist

Open `pyproject.toml` and confirm every line below is present. Tick each:

- [ ] `[build-system].requires = ["hatchling"]`
- [ ] `[build-system].build-backend = "hatchling.build"`
- [ ] `[project].name`, `[project].version`, `[project].dependencies` (with
      `flwr[simulation]`, `flwr-datasets`, `torch`, `torchvision`, `numpy`)
- [ ] `[tool.hatch.build.targets.wheel].packages = ["."]`
- [ ] `[tool.flwr.app].publisher`
- [ ] **`[tool.flwr.app.components].serverapp = "qpriviot_fl.server_app:app"`**
- [ ] **`[tool.flwr.app.components].clientapp = "qpriviot_fl.client_app:app"`**
- [ ] `[tool.flwr.app.config]` block with at least `num-server-rounds`,
      `dataset`, `learning-rate`, `seed`
- [ ] `[tool.flwr.federations].default = "local-simulation"`
- [ ] `[tool.flwr.federations.local-simulation].options.num-supernodes`
- [ ] `[tool.flwr.federations.local-simulation].options.backend.client-resources.num-cpus`

### 3.2 Snapshot-and-diff workflow

Before any structural cleanup, snapshot the current file:

```bash
cp pyproject.toml pyproject.toml.bak
```

After cleanup edits, diff:

```bash
diff -u pyproject.toml.bak pyproject.toml | less
```

If the diff shows any deletion under `[tool.flwr.*]`, restore the line. Once
satisfied:

```bash
rm pyproject.toml.bak
```

### 3.3 Functional verification

These commands will fail loudly if a critical key was deleted:

```bash
# Lint the TOML syntax
python -c "import tomllib; tomllib.loads(open('pyproject.toml','rb').read().decode())" \
  && echo "TOML syntax OK"

# Confirm Flower can resolve the app entry points
python -c "
import tomllib
cfg = tomllib.loads(open('pyproject.toml','rb').read().decode())
comps = cfg['tool']['flwr']['app']['components']
assert 'serverapp' in comps and 'clientapp' in comps, 'components missing'
print('serverapp =', comps['serverapp'])
print('clientapp =', comps['clientapp'])
"

# End-to-end: a 1-round smoke run still works
flwr run . --run-config 'dataset="iot" num-server-rounds=1'
```

### 3.4 Lock the file in CI (optional but recommended)

Add a tiny pre-commit hook so accidental deletions are caught:

```yaml
# .pre-commit-config.yaml (snippet)
- repo: local
  hooks:
    - id: check-flwr-components
      name: pyproject must declare flwr components
      entry: python -c "import tomllib,sys; c=tomllib.loads(open('pyproject.toml','rb').read().decode())['tool']['flwr']['app']['components']; sys.exit(0 if 'serverapp' in c and 'clientapp' in c else 1)"
      language: system
      pass_filenames: false
      files: ^pyproject\.toml$
```

---

## Step 4 — Resource Management (Disk Cleanup)

### 4.1 Inventory before deletion

```bash
# Top-level disk hogs
du -sh * .[!.]* 2>/dev/null | sort -h | tail -20

# Largest tracked files in git history
git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | awk '$1=="blob"' | sort -k3 -n | tail -20
```

Typical offenders in a Flower project:
- `global_model.npz`, `*.pt`, `*.ckpt` — model checkpoints
- `experiment_results/`, `results_*.json` — per-round logs
- `figures/` — PNGs (regenerable from `plot.py`)
- `wandb/`, `ray_results/`, `.flwr/` — runtime caches
- HF dataset cache at `~/.cache/huggingface/`
- `.venv/` — virtualenv (always gitignored)

### 4.2 Archive then delete

Never delete in-place — move first, verify, then drop.

```bash
ARCHIVE="../qpriviot-archive/$(date +%Y%m%d)"
mkdir -p "$ARCHIVE"

# Old simulation logs
mv logs                       "$ARCHIVE/" 2>/dev/null
mv flwr-*.log flwr.log        "$ARCHIVE/" 2>/dev/null

# Old per-round JSON outputs (keep the latest 3 by mtime if you want)
ls -1t experiment_results/results_*.json 2>/dev/null | tail -n +4 \
  | xargs -I {} mv {} "$ARCHIVE/"

# Stale model checkpoints (keep the most recent global_model.npz)
mv *.pt *.pth *.ckpt          "$ARCHIVE/" 2>/dev/null

# Old figures (always regenerable)
mv figures                    "$ARCHIVE/"
mkdir figures                  # placeholder so plot.py can recreate

# Notebook checkpoints
find . -type d -name '.ipynb_checkpoints' -exec rm -rf {} +

# Confirm what was archived
du -sh "$ARCHIVE"/*
```

After verifying you can still rerun experiments, remove the archive:

```bash
rm -rf "$ARCHIVE"
```

### 4.3 Clear runtime caches

These caches are *not* in the repo but eat disk on the host:

```bash
# Pytest / mypy / ruff
rm -rf .pytest_cache .mypy_cache .ruff_cache

# Python bytecode (regenerated automatically)
find . -type d -name '__pycache__' -prune -exec rm -rf {} +
find . -type f -name '*.pyc' -delete

# Flower runtime
rm -rf .flwr .flwr-cache

# Ray (be careful: shared with other projects on the machine)
rm -rf ray_results

# W&B local runs (only after syncing what you need)
wandb sync wandb/ 2>/dev/null   # optional: upload first
rm -rf wandb .wandb

# Hugging Face dataset cache (heaviest — only if you can re-download)
du -sh ~/.cache/huggingface
# rm -rf ~/.cache/huggingface
```

### 4.4 Reclaim space inside `.git`

After untracking large files in Step 2.2, repack the repo to actually shrink:

```bash
# Drop reflogs and run aggressive GC
git reflog expire --expire=now --all
git gc --prune=now --aggressive
du -sh .git
```

> ⚠️ This rewrites local history's recoverable state. Only run after `git
> push` — once collected, dropped objects cannot be recovered.

### 4.5 Verification

```bash
# Disk now and disk before — log both
du -sh . > /tmp/after.txt
git status
flwr run . --run-config 'dataset="iot" num-server-rounds=1'   # still works?
pytest tests/ -m "not slow"                                    # still green?
```

---

## Step 5 — Commit the Cleanup

```bash
git add .gitignore pyproject.toml qpriviot_fl/ scripts/ tests/ docs/
git status                       # eyeball the staged set one last time
git commit -m "chore: repo cleanup — gitignore, layout, prune logs"
git push
git tag -d pre-cleanup-$(date +%Y%m%d)   # only after CI passes
```

---

## Cleanup Cheat-Sheet (One-Page)

| What | Command |
|------|---------|
| Tag safety net | `git tag pre-cleanup-$(date +%Y%m%d)` |
| Verify TOML | `python -c "import tomllib; tomllib.loads(open('pyproject.toml','rb').read().decode())"` |
| Untrack ignored files | `git rm --cached -r experiment_results/ figures/ *.npz` |
| Smoke-test app | `flwr run . --run-config 'dataset="iot" num-server-rounds=1'` |
| Run tests | `pytest tests/ -m "not slow"` |
| Inventory disk | `du -sh * .[!.]* \| sort -h \| tail` |
| Drop pycache | `find . -type d -name __pycache__ -prune -exec rm -rf {} +` |
| Repack git | `git gc --prune=now --aggressive` |