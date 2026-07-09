"""
Phase-2 REAL-DATA leakage-drop probe: MIA / extraction against the DP centroid memory
built from LongMemEval dialogue-turn embeddings (real notes, real user partition).

Same measurement principle as _agentmem_leakage.py, on real data:
  - members    = dialogue turns actually aggregated into the shared memory;
  - non-members= held-out turns (same distribution), never aggregated;
  - MIA score  = cos(turn, its assigned bucket centroid); ROC-AUC (0.5 = no leakage);
  - low-count tail = turns whose bucket has <= T contributors (where a SUM mechanism
    leaks most; DP should protect exactly there);
  - extraction gap = centroid-reconstruction advantage of members over non-members.

Faithful crypto path (real privacy_utils via reused helpers). Multi-seed + mean+/-std.

Usage:
  .venv/bin/python scripts/_longmemeval_leakage.py --K 1024 --d 32 --eps 16,8,3,1 --seeds 0,1
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
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _longmemeval_data import get_embeddings  # noqa: E402
from _agentmem_probe import (  # noqa: E402
    assign_buckets,
    clip_rows_to_norm,
    make_projector,
    single_shot_sigma,
)
from _agentmem_leakage import mia_auc, extraction_gap, build_dp_centroids  # noqa: E402


def run(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    eps_points = [float(e) for e in args.eps.split(",")]
    sigma_specs = [("inf (clean)", 0.0)] + [(f"eps={e:g}", single_shot_sigma(e)) for e in eps_points]
    sigma_specs.append(("FL-sigma", args.fl_sigma))

    print("=== Phase-2 LongMemEval real-data LEAKAGE-DROP probe ===")
    print(f"variant={args.variant} | K={args.K} | proj-d={args.d} | M={args.M} | seeds={seeds}")
    print("MIA-AUC: 0.500 = NO leakage. low-count tail = buckets with few contributors.\n")

    metrics = {lab: {"auc": [], "lc_auc": [], "gap": []} for lab, _ in sigma_specs}
    lc_fracs = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        note_raw, note_user, note_ev, _, _ = get_embeddings(args.variant)
        N = int(note_user.max() + 1)

        note_emb = normalize(
            make_projector(note_raw, args.d, seed, args.proj, public="20ng")(note_raw)).astype(np.float32)

        anchors = normalize(rng.standard_normal((args.K, args.d))).astype(np.float32)
        note_bucket = assign_buckets(note_emb, anchors)

        # split notes: aggregated pool (members) vs held-out non-members (same dist)
        perm = rng.permutation(len(note_emb))
        n_pool = int(0.8 * len(note_emb))
        pool_idx, hold_idx = perm[:n_pool], perm[n_pool:]
        in_pool = np.zeros(len(note_emb), dtype=bool)
        in_pool[pool_idx] = True

        # per-user (real partition) sum-vector + count, from POOL notes only
        user_vecs = [np.zeros((args.K, args.d)) for _ in range(N)]
        user_cnts = [np.zeros(args.K) for _ in range(N)]
        for i in pool_idx:
            u = note_user[i]; b = note_bucket[i]
            user_vecs[u][b] += note_emb[i]; user_cnts[u][b] += 1.0

        C_v = float(np.percentile([np.linalg.norm(v) for v in user_vecs], 95))
        C_c = float(np.percentile([np.linalg.norm(c) for c in user_cnts], 95))
        for u in range(N):
            user_vecs[u] = clip_rows_to_norm(user_vecs[u].reshape(1, -1), C_v).reshape(args.K, args.d)
            user_cnts[u] = clip_rows_to_norm(user_cnts[u].reshape(1, -1), C_c).reshape(args.K)
        B_v = max(float(np.max([np.max(np.abs(v)) for v in user_vecs])), 1e-6)
        B_c = max(float(np.max([np.max(np.abs(c)) for c in user_cnts])), 1e-6)
        sum_c_clean = np.sum(user_cnts, axis=0)
        nonempty = sum_c_clean > 0.5

        # balanced member / non-member samples
        m_sel = rng.choice(pool_idx, size=min(args.M, len(pool_idx)), replace=False)
        n_sel = rng.choice(hold_idx, size=min(args.M, len(hold_idx)), replace=False)
        mem_emb, mem_b = note_emb[m_sel], note_bucket[m_sel]
        non_emb, non_b = note_emb[n_sel], note_bucket[n_sel]
        y_true = np.concatenate([np.ones(len(m_sel)), np.zeros(len(n_sel))])

        T = args.lowcount
        lc_m = sum_c_clean[mem_b] <= T
        lc_n = sum_c_clean[non_b] <= T
        lc_fracs.append(float(np.mean(lc_m)))

        for lab, sigma in sigma_specs:
            cent = build_dp_centroids(user_vecs, user_cnts, args.K, args.d, N,
                                      C_v, C_c, B_v, B_c, sigma, seed, nonempty)
            s_mem = mia_auc(mem_emb, mem_b, cent)
            s_non = mia_auc(non_emb, non_b, cent)
            auc = roc_auc_score(y_true, np.concatenate([s_mem, s_non]))
            if lc_m.sum() >= 10 and lc_n.sum() >= 10:
                y_lc = np.concatenate([np.ones(lc_m.sum()), np.zeros(lc_n.sum())])
                lc_auc = roc_auc_score(y_lc, np.concatenate([s_mem[lc_m], s_non[lc_n]]))
            else:
                lc_auc = float("nan")
            rm, rn = extraction_gap(mem_emb, mem_b, non_emb, non_b, cent, args.K)
            metrics[lab]["auc"].append(float(auc))
            metrics[lab]["lc_auc"].append(float(lc_auc))
            metrics[lab]["gap"].append(rm - rn)

    def ms(xs):
        return float(np.nanmean(xs)), (float(np.nanstd(xs)) if len(xs) > 1 else 0.0)

    print(f"corpus: {N} users, {len(note_emb)} notes; low-count tail <= {args.lowcount} "
          f"contributors ({np.mean(lc_fracs)*100:.0f}% of members)\n")
    print(f"{'release':<14}{'sigma':>7}{'MIA-AUC':>15}{'AUC(low-cnt)':>16}{'extract-gap':>15}")
    print("-" * 67)
    for lab, sigma in sigma_specs:
        a_m, a_s = ms(metrics[lab]["auc"])
        l_m, l_s = ms(metrics[lab]["lc_auc"])
        g_m, g_s = ms(metrics[lab]["gap"])
        print(f"{lab:<14}{sigma:>7.3f}{a_m:>9.3f}+/-{a_s:<4.2f}{l_m:>10.3f}+/-{l_s:<4.2f}{g_m:>9.3f}+/-{g_s:<4.2f}")
    print(f"\nclean AUC(all)={ms(metrics['inf (clean)']['auc'])[0]:.3f}, "
          f"clean tail-AUC={ms(metrics['inf (clean)']['lc_auc'])[0]:.3f}; ideal=0.500.")

    if args.json:
        rows = []
        for lab, sigma in sigma_specs:
            a_m, a_s = ms(metrics[lab]["auc"])
            l_m, l_s = ms(metrics[lab]["lc_auc"])
            g_m, g_s = ms(metrics[lab]["gap"])
            rows.append({"label": lab, "sigma": sigma,
                         "auc_mean": a_m, "auc_std": a_s,
                         "lc_auc_mean": l_m, "lc_auc_std": l_s,
                         "gap_mean": g_m, "gap_std": g_s,
                         "auc_per_seed": [float(x) for x in metrics[lab]["auc"]],
                         "lc_auc_per_seed": [float(x) for x in metrics[lab]["lc_auc"]]})
        out = {"script": "longmemeval_leakage", "metric": "MIA-AUC / tail-AUC / extract-gap",
               "seeds": seeds, "lc_frac": float(np.mean(lc_fracs)),
               "config": {"variant": args.variant, "K": args.K, "d": args.d, "proj": args.proj,
                          "M": args.M, "lowcount": args.lowcount, "eps": args.eps},
               "corpus": {"users": int(N), "notes": int(len(note_emb))}, "rows": rows}
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.json, "w"), indent=2)
        print(f"[json] wrote {args.json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", type=str, default="oracle", choices=["oracle", "s"])
    p.add_argument("--K", type=int, default=1024)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--proj", type=str, default="pca", choices=["pca", "randproj", "publicpca"],
                   help="pca = data-dependent (leaks); randproj = public-seed, zero privacy cost")
    p.add_argument("--M", type=int, default=2000)
    p.add_argument("--lowcount", type=int, default=3)
    p.add_argument("--eps", type=str, default="16,8,3,1")
    p.add_argument("--fl_sigma", type=float, default=2.854)
    p.add_argument("--seeds", type=str, default="0,1")
    p.add_argument("--json", type=str, default=None, help="optional path to dump aggregated metrics")
    run(p.parse_args())