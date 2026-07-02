#!/usr/bin/env python3
"""
_qfl_cnn.py — matched CLASSICAL-CNN baseline for the DP+QFL crossover comparison.

Runs a real convolutional net (large d) through the EXACT SAME federated DP harness as the
VQC (`_qfl_mnist.py`): same flwr DirichletPartitioner(alpha) per-N partitioning, same
per-client cap = n_train//N (the MNIST data-per-client confound), same byte-identical DP path
(quantize / apply_distributed_skellam_noise / dequantize from privacy_utils), same Opacus
sigma calibration, same modes (no-dp plain FedAvg / distributed Skellam-under-SecAgg / local).
The ONLY difference vs the VQC run is the model: a ~50k-param CNN on raw 28x28 images instead
of a d~130-200 VQC on PCA-angle features. This is the high-d endpoint of the d-governs-crossover
comparison, on an IDENTICAL task (same class subset, partition, sigma) so the contrast is fair.

Usage:
  python3 scripts/_qfl_cnn.py --classes 0,1,2,3,4 --Ns 10,50,100,200 --eps 8.0 \
      --rounds 25 --out experiment_results/qfl_cnn_5class_eps8.0.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qpriviot_fl.privacy_utils import (  # noqa: E402
    quantize, dequantize, apply_distributed_skellam_noise,
)
from opacus.accountants.utils import get_noise_multiplier  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--classes", default="0,1,2,3,4")
ap.add_argument("--rounds", type=int, default=25)
ap.add_argument("--local_epochs", type=int, default=4)
ap.add_argument("--local_batch", type=int, default=64)
ap.add_argument("--local_lr", type=float, default=0.05)
ap.add_argument("--eps", type=float, default=8.0)
ap.add_argument("--clip", type=float, default=0.5)
ap.add_argument("--Ns", default="10,50,100,200")
ap.add_argument("--modes", default="no-dp,distributed,local")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--n_train", type=int, default=6000)
ap.add_argument("--n_test", type=int, default=2000)
ap.add_argument("--alpha", type=float, default=0.3)
ap.add_argument("--range_max", type=int, default=1_000_000)
ap.add_argument("--out", default="")
A = ap.parse_args()

CLASSES = [int(c) for c in A.classes.split(",")]
N_CLASSES = len(CLASSES)
RANGE_MAX = A.range_max
torch.manual_seed(A.seed)
np.random.seed(A.seed)
SIGMA = float(get_noise_multiplier(target_epsilon=A.eps, target_delta=1e-5,
                                   sample_rate=1.0, epochs=A.rounds))


def load_partitions(Ns, seed):
    """Same per-N Dirichlet partitioning + class subset + cap=n_train//N as the VQC run, but
    returns raw (1,28,28) image tensors instead of PCA-angle features."""
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import DirichletPartitioner
    from datasets import load_dataset

    rng = np.random.default_rng(seed)
    remap = {c: i for i, c in enumerate(CLASSES)}

    def imgs(batch):
        x = np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in batch]).reshape(-1, 1, 28, 28)
        return (x - 0.1307) / 0.3081  # standard MNIST normalization

    clients_by_N = {}
    for N in Ns:
        part = DirichletPartitioner(num_partitions=N, partition_by="label",
                                    alpha=A.alpha, seed=seed, min_partition_size=10)
        fds = FederatedDataset(dataset="ylecun/mnist", partitioners={"train": part})
        cap = max(20, A.n_train // N)
        clients = []
        for cid in range(N):
            p = fds.load_partition(cid, "train").with_format("numpy")
            lab = np.asarray(p["label"], dtype=np.int64)
            keep = np.where(np.isin(lab, CLASSES))[0]
            if len(keep) == 0:
                clients.append((np.zeros((0, 1, 28, 28), np.float32), np.zeros((0,), np.int64)))
                continue
            if len(keep) > cap:
                keep = rng.choice(keep, cap, replace=False)
            X = imgs(p["image"][keep]); y = np.array([remap[v] for v in lab[keep]], dtype=np.int64)
            clients.append((X.astype(np.float32), y))
        clients_by_N[N] = clients

    ds_test = load_dataset("ylecun/mnist", split="test").with_format("numpy")
    lab = np.asarray(ds_test["label"]); keep = np.where(np.isin(lab, CLASSES))[0]
    if len(keep) > A.n_test:
        keep = rng.choice(keep, A.n_test, replace=False)
    Xte = imgs(ds_test["image"][keep]).astype(np.float32)
    yte = np.array([remap[v] for v in lab[keep]], dtype=np.int64)
    return clients_by_N, Xte, yte


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, N_CLASSES)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.act(self.c1(x)))   # 28 -> 14
        x = self.pool(self.act(self.c2(x)))   # 14 -> 7
        x = x.flatten(1)
        return self.fc2(self.act(self.fc1(x)))


def get_flat(m):
    return torch.cat([p.detach().flatten() for p in m.parameters()])


def set_flat(m, vec):
    i = 0
    for p in m.parameters():
        n = p.numel(); p.data.copy_(vec[i:i + n].view_as(p)); i += n


_PROTO = CNN()
D = int(get_flat(_PROTO).numel())


def evaluate(m, Xte, yte):
    m.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(Xte), 1000):
            preds.append(m(torch.as_tensor(Xte[i:i + 1000])).argmax(1).numpy())
    return float((np.concatenate(preds) == yte).mean())


def local_train(w_global, Xc, yc, epochs, lr, rng):
    m = CNN(); set_flat(m, w_global)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lossfn = nn.CrossEntropyLoss()
    yt = torch.as_tensor(yc, dtype=torch.long)
    n = len(Xc); bs = A.local_batch if A.local_batch and A.local_batch < n else n
    m.train()
    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossfn(m(torch.as_tensor(Xc[idx])), yt[idx])
            loss.backward(); opt.step()
    return (get_flat(m) - w_global).numpy()


def clip_delta(d, C):
    nrm = np.linalg.norm(d)
    return d * min(1.0, C / (nrm + 1e-12))


def run(mode, clients, N, Xte, yte):
    torch.manual_seed(A.seed)
    g = CNN(); w = get_flat(g)
    C = B = A.clip
    best, traj = 0.0, []
    for r in range(A.rounds):
        raw, dq = [], []
        for ci in range(N):
            Xc, yc = clients[ci]
            if len(Xc) == 0:
                continue
            d = local_train(w, Xc, yc, A.local_epochs, A.local_lr, np.random.default_rng(A.seed))
            if mode == "no-dp":
                raw.append(d)
            else:
                d = clip_delta(d, C)
                q = quantize([d], clip_range=B, range_max=RANGE_MAX)
                qn = apply_distributed_skellam_noise(
                    q, sigma=SIGMA, clip_norm=C, num_clients=N, quantization_bound=B,
                    range_max=RANGE_MAX, distributed=(mode == "distributed"),
                    seed=A.seed * 100000 + r * 1000 + ci)
                dq.append(qn[0])
        if mode == "no-dp":
            agg = np.mean(raw, axis=0)
        else:
            agg = dequantize([np.sum(dq, axis=0)], clip_range=B, range_max=RANGE_MAX)[0] / N
        w = w + torch.as_tensor(agg.astype(np.float32))
        set_flat(g, w)
        acc = evaluate(g, Xte, yte); best = max(best, acc); traj.append(round(acc, 4))
    return best, traj


def main():
    Ns = [int(x) for x in A.Ns.split(",")]
    modes = A.modes.split(",")
    t0 = time.time()
    print(f"# QFL-CNN baseline: {N_CLASSES}-class MNIST {CLASSES}, Dirichlet(alpha={A.alpha}), "
          f"CNN d={D} (raw 28x28); eps={A.eps} -> sigma={SIGMA:.4f}, clip C={A.clip}, T={A.rounds}, "
          f"le={A.local_epochs} batch={A.local_batch} lr={A.local_lr}", flush=True)
    clients_by_N, Xte, yte = load_partitions(Ns, A.seed)
    sizes = {N: len(clients_by_N[N][0][0]) for N in Ns}
    print(f"# data/client = n_train//N: {sizes}; test n={len(yte)}; data load {time.time()-t0:.0f}s",
          flush=True)
    results = {"meta": {"d": D, "model": "cnn", "classes": CLASSES, "sigma": SIGMA, "eps": A.eps,
                        "clip": A.clip, "rounds": A.rounds, "alpha": A.alpha, "n_train": A.n_train,
                        "n_test": len(yte), "seed": A.seed}, "runs": {}}
    for mode in modes:
        for n in Ns:
            ts = time.time()
            best, traj = run(mode, clients_by_N[n], n, Xte, yte)
            results["runs"][f"{mode}_N{n}"] = {"best": best, "traj": traj}
            print(f"  {mode:<12} N={n:<4} peak={best*100:5.1f}%  ({time.time()-ts:.0f}s)  "
                  f"traj={[round(a*100) for a in traj]}", flush=True)
            if A.out:
                Path(A.out).parent.mkdir(parents=True, exist_ok=True)
                Path(A.out).write_text(json.dumps(results, indent=2))
    print("\n# DP cost (no-dp - distributed):", flush=True)
    for n in Ns:
        nd = results["runs"].get(f"no-dp_N{n}", {}).get("best")
        di = results["runs"].get(f"distributed_N{n}", {}).get("best")
        if nd is not None and di is not None:
            print(f"  N={n}: {(nd-di)*100:.1f}pp", flush=True)
    if A.out:
        print(f"\n# wrote {A.out}  (total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
