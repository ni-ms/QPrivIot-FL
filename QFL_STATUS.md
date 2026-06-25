# QFL Scale-Up — Working Status (paused 2026-06-24)

Status snapshot for the **DP + Quantum FL** direction (Direction A: "the private-FL
client-count crossover is governed by model dimension d"). All background jobs are STOPPED.
This file is the resume point. Companion: `PAPER_STATUS.md` §8.1/§8.2 (full narrative),
memory `project_qpriviot_fl.md`.

---

## 1. Where we are

Committed Direction A and built a self-contained QFL harness that reuses the project's REAL DP
path (`quantize` / `apply_distributed_skellam_noise` / `dequantize` from `privacy_utils.py`) and
calibrates σ with the SAME Opacus call the server uses. Two phases done; phase 3 (the clean
5-class contrast) is **in progress and was interrupted mid-run**.

### DONE & trustworthy
- **10-class MNIST VQC-vs-CNN crossover (ε=8)** — `experiment_results/qfl_mnist_eps8.0.json`,
  figures `figures/qfl_crossover_vs_N_8.0.png`. Result: VQC (d=208) DP cost 26/16/14/8.6pp at
  N=10/50/100/200 vs CNN (d=262k) 84/14.5/11/29.5pp. Nuanced (VQC ceiling only ~45% on 10-class
  → "usable-private" was only relative). See PAPER_STATUS §8.2.
- **Controlled d-sweep (the cleanest result)** — `experiment_results/qfl_dsweep.json`,
  `figures/qfl_d_governs_crossover.png`. Same task/σ/N=10, vary only d → DP cost rises
  monotonically: d=275→0pp, 1091→2.2, 4355→4.4, 17411→6.7, 69635→7.4pp; anchors VQC d=48→0pp,
  CNN d=262k→84pp.

### IN PROGRESS — Option 3: clean 5-class contrast (interrupted)
Goal: a task where the VQC genuinely learns (~70-85%) so "usable-private VQC where CNN collapses"
becomes an ABSOLUTE claim, killing the "VQC too small to be useful" reviewer critique.

- **VQC 5-class (classes 0-4) — PARTIAL, looking GOOD.** `qfl_mnist_5class_eps8.0.json` has only
  no-dp N=10 (**73.2%**) and N=50 (63.4%) before it was stopped. The ~73% ceiling (vs ~45% at
  10-class) is the whole point — Option 3 is delivering. Still need: no-dp N=100/200,
  distributed N={10,50,100,200} (the headline cells), local N={all}.
- **CNN 5-class — NOT YET VALID.** First attempt (lr=0.05, `cnn5.log`) was BUGGY: no-dp diverged
  to 20% (random) because unclipped Adam deltas explode in plain FedAvg, while the clipped
  distributed path partially learned (36-48%) — i.e. no-dp < distributed, which is impossible and
  flagged the bug. **Fix identified: lr 0.05 → 0.01** (matches the project's own past fix; see
  `FINDINGS`/memory bug #7). The relaunch (lr=0.01, le=3) was killed during data-load before
  producing any cell. **`qfl_cnn_5class_eps8.0.json` currently holds the STALE BUGGED lr=0.05
  data — overwrite it on rerun.**

---

## 2. EXACT resume commands

```bash
cd /Users/macadmin/PycharmProjects/QPrivIot-FL
PY=.venv/bin/python

# (A) Re-run the VQC 5-class sweep to completion (~25-30 min; ceiling already ~73% at N=10):
$PY scripts/_qfl_mnist.py --classes 0,1,2,3,4 --Ns 10,50,100,200 \
    --modes no-dp,distributed,local --rounds 30 --local_epochs 5 --local_batch 64 \
    --local_lr 0.04 --eps 8.0 --alpha 0.3 --n_train 6000 --n_test 2000 \
    --out experiment_results/qfl_mnist_5class_eps8.0.json

# (B) Re-run the CNN 5-class baseline with the FIXED lr (overwrites the stale bugged json):
$PY scripts/_qfl_cnn.py --classes 0,1,2,3,4 --Ns 10,50,100,200 \
    --modes no-dp,distributed,local --rounds 25 --local_epochs 3 --local_batch 64 \
    --local_lr 0.01 --eps 8.0 --alpha 0.3 --n_train 6000 --n_test 2000 \
    --out experiment_results/qfl_cnn_5class_eps8.0.json

# (C) Generate the matched 5-class figures + d-governs panel:
$PY scripts/generate_qfl_figures.py --matched \
    --vqc experiment_results/qfl_mnist_5class_eps8.0.json \
    --cnn experiment_results/qfl_cnn_5class_eps8.0.json --tag 5class_eps8

# Optional later: ε=3 robustness (reuse either harness with --eps 3.0); 10-class ε=3 was
# started then killed (only no-dp cells, which are ε-independent) — qfl_mnist_eps3.0.json is
# incomplete, rerun if wanted.
```

**WATCH-OUT on rerun:** the CNN no-dp must be a HIGH ceiling (~98%) and ≥ its own distributed
result. If CNN no-dp is again stuck ~20%, lr is still too high / training unstable — drop lr
further (0.005) or reduce le. The VQC side is stable; don't touch its hyperparams.

---

## 3. Files created this session

Scripts (`scripts/`):
- `_qfl_mnist.py` — federated VQC runner. Data re-uploading + rich ⟨Z⟩+⟨ZZ⟩ readout + linear
  head (shallow, to dodge the barren plateau). `--classes` subsets+remaps; per-N Dirichlet
  re-partition with cap = n_train//N (replicates the MNIST data-per-client confound); no-dp =
  plain float FedAvg (NO clip), DP modes clip+quantize+Skellam. σ via Opacus.
- `_qfl_cnn.py` — matched classical-CNN baseline (raw 28×28, d≈105k) through the IDENTICAL
  harness (same partition/DP/σ/modes). **Needs lr=0.01 (not 0.05).**
- `_qfl_tune.py` — fast CENTRALIZED VQC tuner (`--classes`). Used to find the shallow+rich
  architecture and confirm ceilings (10-class 53.5%, 5-class 77.4% centralized).
- `_qfl_dsweep.py` — controlled MLP d-sweep on 3-class digits (`--out` writes json).
- `generate_qfl_figures.py` — `--matched` (VQC json vs CNN json) or `--vs-cnn10` (vs project
  262k-CNN result JSONs); writes crossover + DP-cost + d-governs figures.

Results (`experiment_results/`): `qfl_mnist_eps8.0.json` (10-class, complete),
`qfl_dsweep.json` (complete), `qfl_mnist_5class_eps8.0.json` (PARTIAL — 2 cells),
`qfl_cnn_5class_eps8.0.json` (STALE/BUGGED — overwrite), `qfl_mnist_eps3.0.json` (incomplete).
Figures (`figures/`): `qfl_crossover_vs_N_8.0.png`, `qfl_d_governs_crossover.png`.

---

## 4. Key engineering lessons (so we don't re-learn them)

1. **Barren plateau:** deeper VQC trained WORSE. Use SHALLOW (1 entangling layer/block) + data
   re-uploading + rich ⟨Z⟩+⟨ZZ⟩ readout. NQ=8, blocks=2 → d≈128-208.
2. **no-dp must NOT clip** (clipping is a DP-only op). Plain float FedAvg for the ceiling.
3. **Per-client data:** re-partition per N (`num_partitions=N`), cap = n_train//N — do NOT use
   `n_train//max(Ns)` (that starves small-N clients).
4. **CNN local lr = 0.01**, not 0.05 (high lr → unclipped-delta divergence in no-dp FedAvg).
5. VQC sims are slow; running 2-3 sweeps concurrently triples wall-clock. Sweeps checkpoint the
   JSON after every cell, so a kill loses only the in-flight cell.

---

## 5. Open decision (was mid-discussion when paused)

The d-sweep (mechanism) + 5-class VQC (usable ceiling) is the publishable core. Paper is viable
but **mid-tier** (IEEE TQE/QCE/ICASSP/QMI) — two reviewer threats: "confirming known √d DP-ERM
theory" (rebut: first in SecAgg+Skellam+QFL setting) and "VQC too small to be useful" (the 5-class
~73% ceiling is the answer). User leaning toward finishing the 5-class clean contrast (Option 3),
then writing with the d-sweep as the rigorous mechanism panel. Remaining after 5-class completes:
ε=3 robustness, then start writing.
