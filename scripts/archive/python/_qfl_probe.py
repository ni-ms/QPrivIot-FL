#!/usr/bin/env python3
"""
_qfl_probe.py — SCOPING probe for the "DP + Quantum FL" direction.

Question: the classical 260k-param CNN collapsed to ~random under distributed-Skellam
DP-under-SecAgg at N=10 (cross-silo). Does a TINY-d variational quantum circuit (VQC,
d~36 trainable angles) survive the SAME DP mechanism at the SAME noise multiplier sigma —
i.e. does the client-count crossover shift LEFT because d is tiny?

This is a faithfulness-first probe: it imports the PROJECT's real DP functions
(quantize / apply_distributed_skellam_noise / dequantize) so the privacy path is byte-for-byte
the one used in the classical crossover experiments. Only the MODEL changes (VQC vs CNN).

Modes:
  no-dp        : FedAvg, no noise (ceiling for this task/model).
  distributed  : per-client Skellam variance (sigmaCs)^2 / N, summed under SecAgg
                 -> aggregate (sigmaCs)^2 ; averaged-update noise std ~ sigma*C/N.
  local        : naive per-client Skellam variance (sigmaCs)^2 (no 1/N share)
                 -> averaged-update noise std ~ sigma*C/sqrt(N).  (the baseline that
                 never crosses in the classical experiments)

Usage:
  python3 scripts/_qfl_probe.py --rounds 20 --sigma 2.854 --Ns 10,50
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import pennylane as qml
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# the REAL project DP mechanism — identical to the classical crossover runs
from qpriviot_fl.privacy_utils import (  # noqa: E402
    quantize, dequantize, apply_distributed_skellam_noise,
)

ap = argparse.ArgumentParser()
ap.add_argument("--n_qubits", type=int, default=4)
ap.add_argument("--n_layers", type=int, default=3)
ap.add_argument("--classes", default="0,1,2,3")
ap.add_argument("--rounds", type=int, default=20)
ap.add_argument("--local_epochs", type=int, default=8)
ap.add_argument("--local_lr", type=float, default=0.1)
ap.add_argument("--sigma", type=float, default=2.854, help="DP noise multiplier (eps=8 in the classical runs)")
ap.add_argument("--clip", type=float, default=0.5, help="L2 clip norm C on the client update")
ap.add_argument("--Ns", default="10,50")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--range_max", type=int, default=1_000_000)
A = ap.parse_args()

CLASSES = [int(c) for c in A.classes.split(",")]
NQ, NL = A.n_qubits, A.n_layers
RANGE_MAX = A.range_max
torch.manual_seed(A.seed)
np.random.seed(A.seed)

# ----- data: sklearn digits -> PCA to n_qubits features, scale to angles in [-pi, pi] -----
X, y = load_digits(return_X_y=True)
mask = np.isin(y, CLASSES)
X, y = X[mask], y[mask]
remap = {c: i for i, c in enumerate(CLASSES)}
y = np.array([remap[v] for v in y])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=A.seed, stratify=y)
pca = PCA(n_components=NQ, random_state=A.seed).fit(Xtr)
sc = StandardScaler().fit(pca.transform(Xtr))
def feat(Z):
    z = sc.transform(pca.transform(Z))
    return np.clip(z, -3, 3) / 3.0 * np.pi  # -> [-pi, pi]
Xtr_a, Xte_a = feat(Xtr).astype(np.float32), feat(Xte).astype(np.float32)
n_classes = len(CLASSES)
assert NQ >= n_classes, "need n_qubits >= n_classes (one Z-expectation per class logit)"

# ----- VQC model (torch interface, classical-simulable) -----
dev = qml.device("default.qubit", wires=NQ)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(NQ), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(NQ))
    return [qml.expval(qml.PauliZ(i)) for i in range(NQ)]

W_SHAPE = (NL, NQ, 3)
D = int(np.prod(W_SHAPE))  # trainable param count

def forward(Xb, w):
    out = circuit(torch.as_tensor(Xb), w)            # list of (batch,) tensors
    logits = torch.stack(out, dim=1)[:, :n_classes]  # (batch, n_classes)
    return logits

def evaluate(w):
    with torch.no_grad():
        logits = forward(Xte_a, w)
        pred = logits.argmax(1).numpy()
    return float((pred == yte).mean())

def local_train(w_global, Xc, yc, epochs, lr):
    w = w_global.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    lossfn = torch.nn.CrossEntropyLoss()
    yt = torch.as_tensor(yc, dtype=torch.long)
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossfn(forward(Xc, w), yt)
        loss.backward()
        opt.step()
    return (w.detach() - w_global.detach()).numpy()  # delta, shape W_SHAPE

def clip_delta(delta, C):
    n = np.linalg.norm(delta)
    return delta * min(1.0, C / (n + 1e-12)), n

# IID partition of the training set into N shards
def partition(N, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xtr_a))
    return [idx[i::N] for i in range(N)]  # round-robin -> balanced, all classes present

def run(mode, N):
    w = torch.tensor(np.random.default_rng(A.seed).normal(0, 0.1, W_SHAPE), dtype=torch.float32)
    shards = partition(N, A.seed)
    C, B = A.clip, A.clip  # quantization bound = clip norm
    best = 0.0
    traj = []
    for r in range(A.rounds):
        deltas_q = []
        for ci, sh in enumerate(shards):
            if len(sh) == 0:
                continue
            d = local_train(w, Xtr_a[sh], ytr[sh], A.local_epochs, A.local_lr)
            d, _ = clip_delta(d, C)
            if mode == "no-dp":
                deltas_q.append(quantize([d], clip_range=B, range_max=RANGE_MAX)[0])
            else:
                q = quantize([d], clip_range=B, range_max=RANGE_MAX)
                qn = apply_distributed_skellam_noise(
                    q, sigma=A.sigma, clip_norm=C, num_clients=N,
                    quantization_bound=B, range_max=RANGE_MAX,
                    distributed=(mode == "distributed"),
                    seed=A.seed * 100000 + r * 1000 + ci,
                )
                deltas_q.append(qn[0])
        agg_int = np.sum(deltas_q, axis=0)                       # SecAgg sum (masks cancel)
        agg = dequantize([agg_int], clip_range=B, range_max=RANGE_MAX)[0] / N  # average delta
        w = w + torch.as_tensor(agg.astype(np.float32))
        acc = evaluate(w)
        best = max(best, acc)
        traj.append(acc)
    return best, traj

Ns = [int(x) for x in A.Ns.split(",")]
print(f"# QFL probe: {n_classes}-class digits, VQC n_qubits={NQ} n_layers={NL} -> d={D} params "
      f"(vs CNN d~262144); sigma={A.sigma} (eps~8), clip C={A.clip}, T={A.rounds}, "
      f"local_epochs={A.local_epochs} lr={A.local_lr}")
print(f"{'mode':<13}" + "".join(f"N={n:<8}" for n in Ns))
results = {}
for mode in ["no-dp", "distributed", "local"]:
    row = f"{mode:<13}"
    for n in Ns:
        best, traj = run(mode, n)
        results[(mode, n)] = (best, traj)
        row += f"{best*100:5.1f}%   "
    print(row)
print("\n# DP cost (no-dp - distributed):")
for n in Ns:
    print(f"  N={n}: {(results[('no-dp',n)][0]-results[('distributed',n)][0])*100:.1f}pp "
          f"| distributed traj: {[round(a*100) for a in results[('distributed',n)][1]]}")
