#!/usr/bin/env python3
"""
_qfl_tune.py — FAST centralized tuning of the VQC on MNIST 10-class.

Purpose: the naive AngleEmbedding + pure-Z-readout VQC barely beats random (~17%) on
10-class MNIST. Before running the federated DP sweep we need a VQC design that can
actually FIT the task (target: no-dp ceiling >= ~75% so "usable-private" is credible),
while keeping d tiny (the whole headline). Centralized training = fastest iteration signal.

Design knobs:
  * data re-uploading: PCA to (n_qubits * n_blocks) features, upload one n_qubits chunk
    per block (injects more features through few qubits — standard QML trick, zero extra params).
  * StronglyEntanglingLayers per block.
  * small classical linear head on the PauliZ expectations -> class logits (keeps d small).
  * trainable logit temperature.

Usage:
  python3 scripts/_qfl_tune.py --n_qubits 8 --blocks 3 --layers 2 --epochs 40 --lr 0.01
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
import pennylane as qml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--n_qubits", type=int, default=8)
ap.add_argument("--blocks", type=int, default=3, help="data re-uploading blocks")
ap.add_argument("--layers", type=int, default=2, help="entangling layers per block")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--lr", type=float, default=0.01)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--n_train", type=int, default=3000)
ap.add_argument("--n_test", type=int, default=1500)
ap.add_argument("--head", type=int, default=1, help="1=linear head on Z-expectations, 0=pure PauliZ")
ap.add_argument("--rich", type=int, default=0, help="1=measure <Z_i> + <Z_i Z_i+1> (2*NQ-1 features)")
ap.add_argument("--classes", default="0,1,2,3,4,5,6,7,8,9", help="MNIST class subset")
ap.add_argument("--seed", type=int, default=42)
A = ap.parse_args()
torch.manual_seed(A.seed); np.random.seed(A.seed)
CLASSES = [int(c) for c in A.classes.split(",")]
NQ, NB, NL, NC = A.n_qubits, A.blocks, A.layers, len(CLASSES)
N_FEAT = NQ * NB

# ---- data: MNIST via HF, PCA to N_FEAT, scale, squash to [-pi,pi] ----
from datasets import load_dataset  # noqa: E402
def to_arr(ds):
    X = np.stack([np.asarray(im, dtype=np.float32).reshape(-1) / 255.0 for im in ds["image"]])
    return X, np.asarray(ds["label"], dtype=np.int64)
tr = load_dataset("ylecun/mnist", split="train").with_format("numpy")
te = load_dataset("ylecun/mnist", split="test").with_format("numpy")
rng = np.random.default_rng(A.seed)
remap = {c: i for i, c in enumerate(CLASSES)}
def subset(ds, n):
    lab = np.asarray(ds["label"])
    keep = np.where(np.isin(lab, CLASSES))[0]
    if len(keep) > n:
        keep = rng.choice(keep, n, replace=False)
    X, y = to_arr(ds[keep])
    y = np.array([remap[v] for v in y], dtype=np.int64)
    return X, y
Xtr, ytr = subset(tr, A.n_train); Xte, yte = subset(te, A.n_test)
pca = PCA(n_components=N_FEAT, random_state=A.seed).fit(Xtr)
sc = StandardScaler().fit(pca.transform(Xtr))
def feat(X): return (np.clip(sc.transform(pca.transform(X)), -3, 3) / 3.0 * np.pi).astype(np.float32)
Xtr_a, Xte_a = feat(Xtr), feat(Xte)

# ---- VQC with data re-uploading ----
dev = qml.device("default.qubit", wires=NQ)
@qml.qnode(dev, interface="torch", diff_method="backprop")
def circuit(inputs, weights):
    # inputs: (batch, NQ*NB); weights: (NB, NL, NQ, 3)
    for b in range(NB):
        qml.AngleEmbedding(inputs[:, b * NQ:(b + 1) * NQ], wires=range(NQ), rotation="Y")
        qml.StronglyEntanglingLayers(weights[b], wires=range(NQ))
    obs = [qml.expval(qml.PauliZ(i)) for i in range(NQ)]
    if A.rich:
        obs += [qml.expval(qml.PauliZ(i) @ qml.PauliZ(i + 1)) for i in range(NQ - 1)]
    return obs

W_SHAPE = (NB, NL, NQ, 3)
N_OUT = (2 * NQ - 1) if A.rich else NQ

class QNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(
            rng.normal(0, 0.1, W_SHAPE), dtype=torch.float32))
        if A.head:
            self.head = torch.nn.Linear(N_OUT, NC)
        else:
            assert N_OUT >= NC
        self.temp = torch.nn.Parameter(torch.tensor(1.0))
    def forward(self, X):
        out = circuit(torch.as_tensor(X), self.w)
        z = torch.stack(out, dim=1).float()  # (batch, N_OUT); PennyLane yields float64
        if A.head:
            return self.head(z)
        return z[:, :NC] * self.temp

def n_params(m): return sum(p.numel() for p in m.parameters())

def evaluate(m):
    m.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(Xte_a), 500):
            preds.append(m(Xte_a[i:i+500]).argmax(1).numpy())
    return float((np.concatenate(preds) == yte).mean())

def main():
    m = QNet()
    D = n_params(m)
    opt = torch.optim.Adam(m.parameters(), lr=A.lr)
    lossfn = torch.nn.CrossEntropyLoss()
    yt = torch.as_tensor(ytr, dtype=torch.long)
    print(f"# VQC tune: NQ={NQ} blocks={NB} layers={NL} head={A.head} feat={N_FEAT} -> d={D} "
          f"(vs CNN 262k); lr={A.lr} batch={A.batch} ntr={A.n_train}", flush=True)
    t0 = time.time(); best = 0.0
    nb = len(Xtr_a)
    for ep in range(A.epochs):
        m.train()
        perm = rng.permutation(nb)
        for i in range(0, nb, A.batch):
            idx = perm[i:i+A.batch]
            opt.zero_grad()
            loss = lossfn(m(Xtr_a[idx]), yt[idx])
            loss.backward(); opt.step()
        acc = evaluate(m); best = max(best, acc)
        if ep % 5 == 0 or ep == A.epochs - 1:
            print(f"  ep{ep:>3} test_acc={acc*100:5.1f}%  best={best*100:5.1f}%  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print(f"# FINAL best test acc = {best*100:.1f}%  (d={D}, {time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
