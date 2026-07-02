#!/usr/bin/env python3
"""
_qfl_dsweep.py — d-dependence probe for the crossover claim.

Fixes N=10 (cross-silo) and the DP noise multiplier sigma, varies MODEL DIMENSION d
(MLP hidden width), and measures the DP cost (no-dp ceiling - distributed-Skellam) of the
SAME project DP mechanism. Tests the central claim: the client-count crossover is governed
by d — DP cost at fixed N grows with d. The VQC (d~48, cost~0 from _qfl_probe.py) and the
classical CNN (d~262k, cost~84pp from the crossover experiments) are the two endpoints; this
fills the middle with a controlled MLP sweep on the SAME 3-class digits task.

Usage: python3 scripts/_qfl_dsweep.py --N 10 --rounds 30 --hidden 4,16,64,256,1024
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qpriviot_fl.privacy_utils import quantize, dequantize, apply_distributed_skellam_noise  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--classes", default="0,1,2")
ap.add_argument("--N", type=int, default=10)
ap.add_argument("--rounds", type=int, default=30)
ap.add_argument("--local_epochs", type=int, default=10)
ap.add_argument("--local_lr", type=float, default=0.05)
ap.add_argument("--sigma", type=float, default=2.854)
ap.add_argument("--clip", type=float, default=0.5)
ap.add_argument("--hidden", default="4,16,64,256,1024")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default="", help="write JSON rows for generate_qfl_figures.py")
A = ap.parse_args()
RANGE_MAX = 1_000_000
CLASSES = [int(c) for c in A.classes.split(",")]
torch.manual_seed(A.seed); np.random.seed(A.seed)

# data: full 64-dim digits (so wider MLPs have signal), 3 classes
X, y = load_digits(return_X_y=True)
m = np.isin(y, CLASSES); X, y = X[m], y[m]
remap = {c: i for i, c in enumerate(CLASSES)}; y = np.array([remap[v] for v in y])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=A.seed, stratify=y)
sc = StandardScaler().fit(Xtr)
Xtr = sc.transform(Xtr).astype(np.float32); Xte = sc.transform(Xte).astype(np.float32)
n_in, n_out = Xtr.shape[1], len(CLASSES)

class MLP(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(n_in, h), torch.nn.ReLU(), torch.nn.Linear(h, n_out))
    def forward(self, x): return self.net(x)

def get_flat(model): return torch.cat([p.detach().flatten() for p in model.parameters()])
def set_flat(model, vec):
    i = 0
    for p in model.parameters():
        n = p.numel(); p.data.copy_(vec[i:i+n].view_as(p)); i += n

def evaluate(model):
    with torch.no_grad():
        pred = model(torch.as_tensor(Xte)).argmax(1).numpy()
    return float((pred == yte).mean())

def partition(N, seed):
    rng = np.random.default_rng(seed); idx = rng.permutation(len(Xtr))
    return [idx[i::N] for i in range(N)]

def run(h, mode):
    torch.manual_seed(A.seed)
    g = MLP(h); w_global = get_flat(g); D = w_global.numel()
    shards = partition(A.N, A.seed); C = B = A.clip
    best = 0.0
    for r in range(A.rounds):
        deltas_q = []
        for ci, sh in enumerate(shards):
            if len(sh) == 0: continue
            local = MLP(h); set_flat(local, w_global)
            opt = torch.optim.Adam(local.parameters(), lr=A.local_lr)
            lossfn = torch.nn.CrossEntropyLoss(); yt = torch.as_tensor(ytr[sh], dtype=torch.long)
            xb = torch.as_tensor(Xtr[sh])
            for _ in range(A.local_epochs):
                opt.zero_grad(); loss = lossfn(local(xb), yt); loss.backward(); opt.step()
            d = (get_flat(local) - w_global).numpy()
            nrm = np.linalg.norm(d); d = d * min(1.0, C / (nrm + 1e-12))
            if mode == "no-dp":
                deltas_q.append(quantize([d], clip_range=B, range_max=RANGE_MAX)[0])
            else:
                q = quantize([d], clip_range=B, range_max=RANGE_MAX)
                qn = apply_distributed_skellam_noise(q, sigma=A.sigma, clip_norm=C, num_clients=A.N,
                        quantization_bound=B, range_max=RANGE_MAX, distributed=(mode == "distributed"),
                        seed=A.seed*100000 + r*1000 + ci)
                deltas_q.append(qn[0])
        agg = dequantize([np.sum(deltas_q, axis=0)], clip_range=B, range_max=RANGE_MAX)[0] / A.N
        w_global = w_global + torch.as_tensor(agg.astype(np.float32))
        set_flat(g, w_global); best = max(best, evaluate(g))
    return best, D

print(f"# d-sweep: 3-class digits, MLP, N={A.N}, sigma={A.sigma} (eps~8), C={A.clip}, T={A.rounds}")
print(f"{'hidden':>7} {'d':>8} {'no-dp':>8} {'distrib':>8} {'DP cost':>8}")
rows = []
for h in [int(x) for x in A.hidden.split(",")]:
    nd, D = run(h, "no-dp"); di, _ = run(h, "distributed")
    print(f"{h:>7} {D:>8} {nd*100:>7.1f}% {di*100:>7.1f}% {(nd-di)*100:>7.1f}pp")
    rows.append({"hidden": h, "d": D, "no_dp": nd, "distributed": di, "dp_cost": nd - di})
print("# anchors: VQC d=48 -> ~0pp (see _qfl_probe.py); CNN d=262144 -> ~84pp (classical crossover, N=10)")
if A.out:
    import json
    from pathlib import Path
    Path(A.out).parent.mkdir(parents=True, exist_ok=True)
    Path(A.out).write_text(json.dumps({"meta": {"N": A.N, "sigma": A.sigma, "rounds": A.rounds,
                                                 "task": "3-class-digits", "model": "MLP"},
                                        "rows": rows}, indent=2))
    print(f"# wrote {A.out}")
