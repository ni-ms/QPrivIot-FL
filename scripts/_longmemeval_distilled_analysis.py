"""
Compare LLM-DISTILLED memory notes vs RAW dialogue turns under the SAME DP centroid-memory
leakage measurement. Answers: does distilling turns into A-MEM/Mem0-style notes (the real agent
setup) change the privacy story?

Leakage (MIA-AUC, low-count-tail AUC, extraction gap) needs no per-turn labels, so it's the clean
distilled-vs-raw comparison. Both sources run the identical crypto path + attack.

Requires the distilled cache from _longmemeval_distill.py and the raw ST cache from
_longmemeval_data.py. Local + free.

Usage:
  .venv/bin/python scripts/_longmemeval_distilled_analysis.py --distilled experiment_results/_lme_distilled_oracle_pilot.json --K 256 --seeds 0,1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _longmemeval_data import get_embeddings as get_raw_embeddings  # noqa: E402
from _agentmem_probe import assign_buckets, clip_rows_to_norm, single_shot_sigma  # noqa: E402
from _agentmem_leakage import mia_auc, extraction_gap, build_dp_centroids  # noqa: E402

_CACHE_DIR = Path(__file__).resolve().parents[1] / "experiment_results"


def embed_distilled(distilled_path):
    """Load distilled notes, embed with cached all-MiniLM-L6-v2, cache to npz."""
    cache = _CACHE_DIR / (Path(distilled_path).stem + "_emb.npz")
    if cache.exists():
        z = np.load(cache)
        return z["emb"], z["user"]
    data = json.load(open(distilled_path))
    texts, users = [], []
    for rec in data:
        for note in rec["notes"]:
            texts.append(note); users.append(rec["user"])
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-MiniLM-L6-v2")
    emb = m.encode(texts, batch_size=256, show_progress_bar=True,
                   normalize_embeddings=False).astype(np.float32)
    np.savez(cache, emb=emb, user=np.array(users))
    return emb, np.array(users)


def run_leakage(note_raw, note_user, K, d, seeds, eps_points, M, lowcount, fl_sigma):
    sigma_specs = [("inf", 0.0)] + [(f"eps={e:g}", single_shot_sigma(e)) for e in eps_points]
    sigma_specs.append(("FL", fl_sigma))
    metrics = {lab: {"auc": [], "lc": [], "gap": []} for lab, _ in sigma_specs}
    lc_fracs = []
    N = int(note_user.max() + 1)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        if d < note_raw.shape[1]:
            note_emb = normalize(PCA(n_components=d, random_state=seed).fit_transform(note_raw)).astype(np.float32)
        else:
            note_emb = normalize(note_raw).astype(np.float32)
        anchors = normalize(rng.standard_normal((K, d))).astype(np.float32)
        bucket = assign_buckets(note_emb, anchors)
        perm = rng.permutation(len(note_emb)); npool = int(0.8 * len(note_emb))
        pool, hold = perm[:npool], perm[npool:]
        uv = [np.zeros((K, d)) for _ in range(N)]; uc = [np.zeros(K) for _ in range(N)]
        for i in pool:
            uv[note_user[i]][bucket[i]] += note_emb[i]; uc[note_user[i]][bucket[i]] += 1.0
        C_v = float(np.percentile([np.linalg.norm(v) for v in uv], 95))
        C_c = float(np.percentile([np.linalg.norm(c) for c in uc], 95))
        for u in range(N):
            uv[u] = clip_rows_to_norm(uv[u].reshape(1, -1), C_v).reshape(K, d)
            uc[u] = clip_rows_to_norm(uc[u].reshape(1, -1), C_c).reshape(K)
        B_v = max(float(np.max([np.max(np.abs(v)) for v in uv])), 1e-6)
        B_c = max(float(np.max([np.max(np.abs(c)) for c in uc])), 1e-6)
        sum_c = np.sum(uc, axis=0); nonempty = sum_c > 0.5
        m_sel = rng.choice(pool, size=min(M, len(pool)), replace=False)
        n_sel = rng.choice(hold, size=min(M, len(hold)), replace=False)
        me, mb = note_emb[m_sel], bucket[m_sel]; ne, nb = note_emb[n_sel], bucket[n_sel]
        y = np.concatenate([np.ones(len(m_sel)), np.zeros(len(n_sel))])
        lc_m = sum_c[mb] <= lowcount; lc_n = sum_c[nb] <= lowcount
        lc_fracs.append(float(np.mean(lc_m)))
        for lab, sigma in sigma_specs:
            cent = build_dp_centroids(uv, uc, K, d, N, C_v, C_c, B_v, B_c, sigma, seed, nonempty)
            sm, sn = mia_auc(me, mb, cent), mia_auc(ne, nb, cent)
            metrics[lab]["auc"].append(roc_auc_score(y, np.concatenate([sm, sn])))
            if lc_m.sum() >= 10 and lc_n.sum() >= 10:
                ylc = np.concatenate([np.ones(lc_m.sum()), np.zeros(lc_n.sum())])
                metrics[lab]["lc"].append(roc_auc_score(ylc, np.concatenate([sm[lc_m], sn[lc_n]])))
            rm, rn = extraction_gap(me, mb, ne, nb, cent, K)
            metrics[lab]["gap"].append(rm - rn)
    return sigma_specs, metrics, float(np.mean(lc_fracs)), N


def print_table(title, sigma_specs, metrics, lc_frac, N, n_notes):
    print(f"\n### {title}: {N} users, {n_notes} notes, low-count tail={lc_frac*100:.0f}% of members")
    print(f"{'release':<10}{'MIA-AUC':>12}{'tail-AUC':>12}{'extract-gap':>14}")
    print("-" * 48)
    for lab, _ in sigma_specs:
        a = np.mean(metrics[lab]["auc"])
        l = np.mean(metrics[lab]["lc"]) if metrics[lab]["lc"] else float("nan")
        g = np.mean(metrics[lab]["gap"])
        print(f"{lab:<10}{a:>12.3f}{l:>12.3f}{g:>14.3f}")


def run(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    eps = [float(e) for e in args.eps.split(",")]

    d_emb, d_user = embed_distilled(args.distilled)
    n_users = int(d_user.max() + 1)

    # raw turns for the SAME users (subset the cached raw embeddings)
    r_emb, r_user, _, _, _ = get_raw_embeddings("oracle")
    mask = r_user < n_users
    r_emb, r_user = r_emb[mask], r_user[mask]

    print(f"=== Distilled vs Raw leakage (K={args.K}, d={args.d}, {n_users} users) ===")
    ss, m, lf, N = run_leakage(r_emb, r_user, args.K, args.d, seeds, eps, args.M, args.lowcount, args.fl_sigma)
    print_table("RAW dialogue turns", ss, m, lf, N, len(r_emb))
    ss, m, lf, N = run_leakage(d_emb, d_user, args.K, args.d, seeds, eps, args.M, args.lowcount, args.fl_sigma)
    print_table("LLM-DISTILLED notes", ss, m, lf, N, len(d_emb))
    print("\n(MIA-AUC / tail-AUC: 0.5 = no leakage. Does DP drive both sources to ~0.5?)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--distilled", type=str, required=True)
    p.add_argument("--K", type=int, default=256)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--M", type=int, default=1000)
    p.add_argument("--lowcount", type=int, default=3)
    p.add_argument("--eps", type=str, default="16,8,3")
    p.add_argument("--fl_sigma", type=float, default=2.854)
    p.add_argument("--seeds", type=str, default="0,1")
    run(p.parse_args())