"""Occupancy detectability on the REAL LongMemEval-oracle corpus, measured with AUC
(F1 is base-rate-gamed: 'predict all occupied' scores 0.97 at K=1024)."""
import sys, math, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from qpriviot_fl.privacy_utils import quantize, apply_distributed_skellam_noise, dequantize

DELTA, QB, RM = 1e-5, 10.0, 1_000_000
SCALE = RM / QB


def rdp(a, s, dim, clip=1.0, sc=SCALE):
    mu = (s * clip * sc) ** 2; d2 = clip * sc; d1 = math.sqrt(dim) * d2
    return a * d2 * d2 / (2 * mu) + min(((2 * a - 1) * d2 * d2 + 6 * d1) / (4 * mu * mu), 3 * d1 / (2 * mu))


def eps_of(ch):
    return min(sum(rdp(a, s, dm) for s, dm in ch) + math.log(1 / DELTA) / (a - 1) for a in range(2, 4097))


def release(per_user, sigma, clip, N):
    Bq = max(float(np.abs(per_user).max()), 1e-6); agg = None
    for u in range(N):
        q = quantize([per_user[u]], clip_range=Bq, range_max=RM)
        q = apply_distributed_skellam_noise(q, sigma=sigma, clip_norm=clip, num_clients=N,
                                            quantization_bound=Bq, range_max=RM, distributed=True,
                                            seed=(u * 104729 + 5) & 0x7FFFFFFF)
        agg = q[0] if agg is None else agg + q[0]
    return dequantize([agg], clip_range=Bq, range_max=RM)[0]


z = np.load("experiment_results/_st_lme_oracle.npz", allow_pickle=True)
E_raw, owner = z["note_emb"], z["note_user"]
N = int(owner.max()) + 1; d = 32
E = PCA(n_components=d, random_state=0).fit_transform(E_raw)
E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
sig_v = 0.606
rng0 = np.random.default_rng(0)

# ── (a) how often are buckets empty?  (paper §7.10a, 5 anchor seeds) ──────────
print("(a) empty-bucket % across 5 anchor seeds; b_max = buckets touched by busiest user\n")
print(f"{'K':>6}{'empty% mean+-sd':>20}{'b_max':>8}")
for K in [32, 64, 128, 256, 512, 1024, 2048]:
    es, bs = [], []
    for sd in range(5):
        r = np.random.default_rng(sd)
        a = r.normal(size=(K, d)); a /= np.linalg.norm(a, axis=1, keepdims=True)
        b = np.argmax(E @ a.T, axis=1)
        es.append(100 * (np.bincount(b, minlength=K) == 0).mean())
        M = np.zeros((N, K), dtype=bool); M[owner, b] = True
        bs.append(int(M.sum(1).max()))
    print(f"{K:>6}{np.mean(es):>13.2f} +- {np.std(es):<5.2f}{max(bs):>8}")

# ── (b) can occupancy be detected privately?  (paper §7.10b) ─────────────────
print("\n(b) Occupancy DETECTABILITY (AUC: empty vs occupied). 0.5 = undetectable.")
print("eps_v-only(sigma=0.606) = 9.284\n")
print("channels: norm = ‖S[k]‖ vs floor (free); cnt = raw counts; ind = binary user-indicator")
print(f"{'K':>6}{'empty%':>8}{'C_c cnt':>9}{'C_c ind':>9}{'norm':>7}"
      f"{'cnt@sv':>8}{'ind@sv':>8}{'eps':>7}{'cnt@2':>8}{'ind@2':>8}{'eps':>7}")
for K in [512, 1024, 2048]:
    anch = rng0.normal(size=(K, d)); anch /= np.linalg.norm(anch, axis=1, keepdims=True)
    b = np.argmax(E @ anch.T, axis=1)
    V = np.zeros((N, K, d)); Cn = np.zeros((N, K))
    for i in range(len(E)):
        V[owner[i], b[i]] += E[i]; Cn[owner[i], b[i]] += 1
    true_c = Cn.sum(0); occ = (true_c > 0.5).astype(int)
    nv = np.linalg.norm(V.reshape(N, -1), axis=1); C_v = float(np.percentile(nv, 95))
    nc = np.linalg.norm(Cn, axis=1); C_c = float(np.percentile(nc, 95))
    Ind = (Cn > 0).astype(float)
    ni = np.linalg.norm(Ind, axis=1); C_i = float(np.percentile(ni, 95))

    Vc = (V.reshape(N, -1) * np.minimum(1.0, C_v / (nv[:, None] + 1e-12))).reshape(N, K, d)
    S = release(Vc, sig_v, C_v, N)
    auc_norm = roc_auc_score(occ, np.linalg.norm(S, axis=1))

    Cc = Cn * np.minimum(1.0, C_c / (nc[:, None] + 1e-12))
    Ic = Ind * np.minimum(1.0, C_i / (ni[:, None] + 1e-12))
    cnt, ind, eps = {}, {}, {}
    for sc in [0.606, 2.0]:
        cnt[sc] = roc_auc_score(occ, release(Cc, sc, C_c, N))
        ind[sc] = roc_auc_score(occ, release(Ic, sc, C_i, N))
        eps[sc] = eps_of([(sig_v, K * d), (sc, K)])
    print(f"{K:>6}{100*(1-occ.mean()):>8.2f}{C_c:>9.2f}{C_i:>9.2f}{auc_norm:>7.3f}"
          f"{cnt[0.606]:>8.3f}{ind[0.606]:>8.3f}{eps[0.606]:>7.2f}"
          f"{cnt[2.0]:>8.3f}{ind[2.0]:>8.3f}{eps[2.0]:>7.2f}")

print("\nNOTE: 'norm' is free (post-processing of the already-released S); 'cnt'/'ind' each cost")
print("      a second RDP composition. Under USER-level DP the indicator channel has L2")
print("      sensitivity sqrt(b_u) (measured C_c ind ~5), NOT 1 -- but it beats raw counts.")
