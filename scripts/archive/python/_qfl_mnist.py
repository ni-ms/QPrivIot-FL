#!/usr/bin/env python3
"""
_qfl_mnist.py — SCALED-UP "DP + Quantum FL" experiment (the fused-paper headline run).

Scales the 3-class-digits scoping probe (`_qfl_probe.py`) to the real crossover task:
MNIST 10-class, Dirichlet(alpha=0.3) non-IID partitioning, SAME DP mechanism and SAME
Opacus-calibrated sigma as the classical CNN crossover runs — so the ONLY thing that
changes vs the CNN is the model (a tiny-d variational quantum circuit instead of a 262k
CNN). Headline figure: the client-count crossover SHIFTS LEFT for the tiny-d VQC — it is
usable-private at N=10 (cross-silo) exactly where the CNN collapsed to ~random.

Faithfulness guarantees:
  * DP path is byte-identical: imports quantize / apply_distributed_skellam_noise /
    dequantize from the project's privacy_utils. The clipped update DELTA (quantum angles
    + classical readout head, one flat vector of dim d) is what gets clipped/quantized/noised.
  * sigma is calibrated with the SAME Opacus call the server uses
    (get_noise_multiplier(eps, 1e-5, sample_rate=1.0, epochs=T)).
  * Partitioning is the SAME flwr DirichletPartitioner(alpha=0.3, partition_by="label", seed)
    used by task.py for the CNN MNIST runs.

VQC architecture (tuned in _qfl_tune.py; shallow to avoid barren plateaus):
  * data re-uploading: PCA to (n_qubits * blocks) features, upload one n_qubits chunk per
    block (injects more features through few qubits, zero extra params).
  * 1 StronglyEntanglingLayer per block (shallow — deeper trained WORSE: barren plateau).
  * rich readout: <Z_i> for all qubits + <Z_i Z_i+1> correlations (2*NQ-1 observables).
  * small classical linear head on those observables -> 10 class logits.
  d = quantum angles + head ≈ 209 (vs CNN 262144). Intrinsic model difference vs CNN
  (images PCA-reduced to angle features) is declared honestly in the paper.

Modes (identical semantics to the probe / classical runs):
  no-dp        FedAvg, no noise (data-only ceiling for this model/task).
  distributed  per-client Skellam variance (sigma*C*s)^2 / N, summed under SecAgg ->
               aggregate variance (sigma*C*s)^2 (central-DP-equiv; mean noise ~ sigma*C/N).
  local        naive per-client variance (sigma*C*s)^2 (mean noise ~ sigma*C/sqrt(N)).

Usage:
  python3 scripts/_qfl_mnist.py --Ns 10,50,100 --eps 8.0 --rounds 30 --out experiment_results/qfl_mnist_eps8.0.json
  python3 scripts/_qfl_mnist.py --smoke
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import pennylane as qml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# the REAL project DP mechanism — byte-identical to the classical crossover runs
from qpriviot_fl.privacy_utils import (  # noqa: E402
    quantize, dequantize, apply_distributed_skellam_noise,
)

ap = argparse.ArgumentParser()
ap.add_argument("--n_qubits", type=int, default=8)
ap.add_argument("--blocks", type=int, default=2, help="data re-uploading blocks (= PCA feats / n_qubits)")
ap.add_argument("--n_layers", type=int, default=1, help="StronglyEntanglingLayers per block (shallow!)")
ap.add_argument("--rich", type=int, default=1, help="1=rich readout <Z_i>+<Z_iZ_i+1>")
ap.add_argument("--rounds", type=int, default=30)
ap.add_argument("--local_epochs", type=int, default=8)
ap.add_argument("--local_batch", type=int, default=32, help="mini-batch size for local SGD (0=full batch)")
ap.add_argument("--local_lr", type=float, default=0.03)
ap.add_argument("--eps", type=float, default=8.0, help="target epsilon; sigma calibrated via Opacus")
ap.add_argument("--clip", type=float, default=0.5, help="L2 clip norm C on the client update (= quantization bound)")
ap.add_argument("--Ns", default="10,50,100")
ap.add_argument("--modes", default="no-dp,distributed,local")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--n_train", type=int, default=6000, help="subsampled global train pool (sim tractability)")
ap.add_argument("--n_test", type=int, default=2000, help="global test pool for evaluation")
ap.add_argument("--alpha", type=float, default=0.3, help="Dirichlet non-IID concentration (matches CNN runs)")
ap.add_argument("--classes", default="0,1,2,3,4,5,6,7,8,9", help="MNIST class subset (remapped to 0..k-1)")
ap.add_argument("--range_max", type=int, default=1_000_000)
ap.add_argument("--out", default="")
ap.add_argument("--smoke", action="store_true", help="tiny run to time the quantum sim")
A = ap.parse_args()

if A.smoke:
    A.Ns, A.rounds, A.n_train, A.n_test = "10", 3, 2000, 800

NQ, NB, NL = A.n_qubits, A.blocks, A.n_layers
N_FEAT = NQ * NB
N_OUT = (2 * NQ - 1) if A.rich else NQ
RANGE_MAX = A.range_max
CLASSES = [int(c) for c in A.classes.split(",")]
N_CLASSES = len(CLASSES)
torch.manual_seed(A.seed)
np.random.seed(A.seed)

# ---- sigma: calibrated with the SAME Opacus call the classical server uses ----
from opacus.accountants.utils import get_noise_multiplier  # noqa: E402
SIGMA = float(get_noise_multiplier(target_epsilon=A.eps, target_delta=1e-5,
                                   sample_rate=1.0, epochs=A.rounds))


# ---- data: SAME flwr Dirichlet(alpha) partitioning of ylecun/mnist as task.py ----
def load_mnist_partitions(Ns, seed):
    """Return ({N: [(X_angles, y) per client]}, Xte_angles, yte).

    Re-partitions MNIST SEPARATELY for each N (DirichletPartitioner(num_partitions=N), exactly
    like the CNN runs) and caps each client to n_train//N samples. So data-per-client DECREASES
    with N (n_train//N) — this faithfully replicates the MNIST fixed-total-data confound (more
    clients => less data each) that produced the classical "window". PCA is fit ONCE on a global
    sample so the featurizer is identical across all N.
    """
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import DirichletPartitioner
    from datasets import load_dataset

    rng = np.random.default_rng(seed)

    def imgs_to_arr(batch_imgs):
        return np.stack([np.asarray(im, dtype=np.float32).reshape(-1) / 255.0 for im in batch_imgs])

    # global PCA/scaler fit on a fixed sample of the chosen classes (N-independent)
    raw_train = load_dataset("ylecun/mnist", split="train").with_format("numpy")
    _lab = np.asarray(raw_train["label"], dtype=np.int64)
    _pool = np.where(np.isin(_lab, CLASSES))[0]
    fit_idx = rng.choice(_pool, min(A.n_train, len(_pool)), replace=False)
    fitX = imgs_to_arr(raw_train[fit_idx]["image"])
    pca = PCA(n_components=N_FEAT, random_state=seed).fit(fitX)
    sc = StandardScaler().fit(pca.transform(fitX))

    def feat(Xraw):
        z = sc.transform(pca.transform(Xraw))
        return (np.clip(z, -3, 3) / 3.0 * np.pi).astype(np.float32)

    remap = {c: i for i, c in enumerate(CLASSES)}
    clients_by_N = {}
    for N in Ns:
        part = DirichletPartitioner(num_partitions=N, partition_by="label",
                                    alpha=A.alpha, seed=seed, min_partition_size=10)
        fds = FederatedDataset(dataset="ylecun/mnist", partitioners={"train": part})
        cap = max(20, A.n_train // N)   # data-per-client decreases with N (the MNIST confound)
        clients = []
        for cid in range(N):
            p = fds.load_partition(cid, "train").with_format("numpy")
            ylab = np.asarray(p["label"], dtype=np.int64)
            keep = np.where(np.isin(ylab, CLASSES))[0]   # subset to chosen classes
            if len(keep) > cap:
                keep = rng.choice(keep, cap, replace=False)
            if len(keep) == 0:
                clients.append((np.zeros((0, N_FEAT), np.float32), np.zeros((0,), np.int64)))
                continue
            X = imgs_to_arr(p["image"][keep])
            y = np.array([remap[v] for v in ylab[keep]], dtype=np.int64)
            clients.append((feat(X), y))
        clients_by_N[N] = clients

    ds_test = load_dataset("ylecun/mnist", split="test").with_format("numpy")
    ylab = np.asarray(ds_test["label"], dtype=np.int64)
    keep = np.where(np.isin(ylab, CLASSES))[0]
    if len(keep) > A.n_test:
        keep = rng.choice(keep, A.n_test, replace=False)
    Xte_raw = imgs_to_arr(ds_test["image"][keep])
    yte = np.array([remap[v] for v in ylab[keep]], dtype=np.int64)
    return clients_by_N, feat(Xte_raw), yte


# ---- VQC model (data re-uploading + rich readout + linear head; torch interface) ----
dev = qml.device("default.qubit", wires=NQ)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def circuit(inputs, weights):
    for b in range(NB):
        qml.AngleEmbedding(inputs[:, b * NQ:(b + 1) * NQ], wires=range(NQ), rotation="Y")
        qml.StronglyEntanglingLayers(weights[b], wires=range(NQ))
    obs = [qml.expval(qml.PauliZ(i)) for i in range(NQ)]
    if A.rich:
        obs += [qml.expval(qml.PauliZ(i) @ qml.PauliZ(i + 1)) for i in range(NQ - 1)]
    return obs


W_SHAPE = (NB, NL, NQ, 3)


class QNet(torch.nn.Module):
    def __init__(self, rng):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(rng.normal(0, 0.1, W_SHAPE), dtype=torch.float32))
        self.head = torch.nn.Linear(N_OUT, N_CLASSES)

    def forward(self, X):
        out = circuit(torch.as_tensor(X), self.w)
        z = torch.stack(out, dim=1).float()
        return self.head(z)


def get_flat(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def set_flat(model, vec):
    i = 0
    for p in model.parameters():
        n = p.numel(); p.data.copy_(vec[i:i + n].view_as(p)); i += n


_PROTO = QNet(np.random.default_rng(0))
D = int(get_flat(_PROTO).numel())  # full trainable dim (quantum angles + head)


def evaluate(model, Xte, yte):
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(Xte), 1000):
            preds.append(model(Xte[i:i + 1000]).argmax(1).numpy())
    return float((np.concatenate(preds) == yte).mean())


def local_train(w_global, Xc, yc, epochs, lr, rng):
    m = QNet(rng); set_flat(m, w_global)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lossfn = torch.nn.CrossEntropyLoss()
    yt = torch.as_tensor(yc, dtype=torch.long)
    n = len(Xc)
    bs = A.local_batch if A.local_batch and A.local_batch < n else n  # mini-batch SGD
    m.train()
    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossfn(m(Xc[idx]), yt[idx])
            loss.backward()
            opt.step()
    return (get_flat(m) - w_global).numpy()


def clip_delta(delta, C):
    n = np.linalg.norm(delta)
    return delta * min(1.0, C / (n + 1e-12))


def run(mode, clients, N, Xte, yte):
    g = QNet(np.random.default_rng(A.seed))
    w = get_flat(g)
    C = B = A.clip
    best, traj = 0.0, []
    for r in range(A.rounds):
        raw_deltas, deltas_q = [], []   # raw_deltas for no-dp (plain FedAvg); deltas_q for DP path
        for ci in range(N):
            Xc, yc = clients[ci]
            if len(Xc) == 0:
                continue
            d = local_train(w, Xc, yc, A.local_epochs, A.local_lr, np.random.default_rng(A.seed))
            if mode == "no-dp":
                # no-dp = plain float FedAvg: NO clip, NO quantize (both are SecAgg/DP-only ops).
                raw_deltas.append(d)
            else:
                d = clip_delta(d, C)   # per-client L2 clip (the DP sensitivity bound)
                q = quantize([d], clip_range=B, range_max=RANGE_MAX)
                qn = apply_distributed_skellam_noise(
                    q, sigma=SIGMA, clip_norm=C, num_clients=N,
                    quantization_bound=B, range_max=RANGE_MAX,
                    distributed=(mode == "distributed"),
                    seed=A.seed * 100000 + r * 1000 + ci,
                )
                deltas_q.append(qn[0])
        if mode == "no-dp":
            agg = np.mean(raw_deltas, axis=0)
        else:
            agg = dequantize([np.sum(deltas_q, axis=0)], clip_range=B, range_max=RANGE_MAX)[0] / N
        w = w + torch.as_tensor(agg.astype(np.float32))
        set_flat(g, w)
        acc = evaluate(g, Xte, yte)
        best = max(best, acc)
        traj.append(round(acc, 4))
    return best, traj


def main():
    Ns = [int(x) for x in A.Ns.split(",")]
    modes = A.modes.split(",")
    t0 = time.time()
    print(f"# QFL-MNIST: {N_CLASSES}-class {CLASSES}, Dirichlet(alpha={A.alpha}), VQC NQ={NQ} blocks={NB} layers={NL} "
          f"rich={A.rich} feat={N_FEAT} -> d={D} (vs CNN d~262144); eps={A.eps} -> sigma={SIGMA:.4f}, "
          f"clip C={A.clip}, T={A.rounds}, le={A.local_epochs}, lr={A.local_lr}", flush=True)
    clients_by_N, Xte, yte = load_mnist_partitions(Ns, A.seed)
    sizes = {N: len(clients_by_N[N][0][0]) for N in Ns}
    print(f"# re-partitioned per N (data/client = n_train//N): {sizes}; test n={len(yte)}; "
          f"data load {time.time()-t0:.0f}s", flush=True)

    results = {"meta": {"d": D, "n_qubits": NQ, "blocks": NB, "n_layers": NL, "rich": A.rich,
                        "classes": CLASSES, "sigma": SIGMA, "eps": A.eps, "clip": A.clip,
                        "rounds": A.rounds, "local_epochs": A.local_epochs, "local_lr": A.local_lr,
                        "alpha": A.alpha, "n_train": A.n_train, "n_test": len(yte),
                        "seed": A.seed}, "runs": {}}
    for mode in modes:
        for n in Ns:
            ts = time.time()
            best, traj = run(mode, clients_by_N[n], n, Xte, yte)
            results["runs"][f"{mode}_N{n}"] = {"best": best, "traj": traj}
            print(f"  {mode:<12} N={n:<4} peak={best*100:5.1f}%  ({time.time()-ts:.0f}s)  "
                  f"traj={[round(a*100) for a in traj]}", flush=True)
            if A.out:  # checkpoint after every cell (sweep is long)
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
