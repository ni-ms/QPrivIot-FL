"""
Phase-2 REAL-DATA utility probe: DP centroid memory on LongMemEval, evaluated by
EVIDENCE RECALL (real ground truth) instead of a topic proxy.

Setup (see _longmemeval_data.py): user = question haystack, note = dialogue turn,
query = question, GT = has_answer turns. Notes from ALL 500 users are pooled into a
shared DP centroid memory under SecAgg+Skellam; for each query we route to the top-k
centroids and check whether the query's own evidence turn's bucket is retrieved.

  evidence-recall@k = mean over queries with evidence of
      [ fraction of the query's evidence turns whose bucket is in the query's top-k centroids ]

Reported clean (no DP) vs DP over an eps sweep, at a chosen projection dim d and #buckets K.
Faithful crypto path (real privacy_utils via _agentmem_probe helpers). Multi-seed.

Usage:
  .venv/bin/python scripts/_longmemeval_probe.py --K 64 --d 32 --topk 5 --eps 16,8,3 --seeds 0,1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _longmemeval_data import get_embeddings  # noqa: E402
from _agentmem_probe import (  # noqa: E402
    assign_buckets,
    clip_rows_to_norm,
    secagg_skellam_release,
    single_shot_sigma,
)


def build_centroids(user_vecs, user_cnts, K, d, N, C_v, C_c, B_v, B_c, sigma, seed, nonempty):
    """SEPARATE releases: sum-vector and counts noised independently (2 RDP compositions)."""
    if sigma == 0.0:
        sv = np.sum(user_vecs, axis=0); sc = np.sum(user_cnts, axis=0)
    else:
        sv = secagg_skellam_release([[v] for v in user_vecs], sigma, C_v, B_v, N, seed)[0]
        sc = secagg_skellam_release([[c.reshape(K, 1)] for c in user_cnts], sigma, C_c, B_c, N, seed)[0].reshape(K)
    sc = np.maximum(sc, 1.0)
    cent = np.zeros((K, d), dtype=np.float32)
    cent[nonempty] = (sv[nonempty] / sc[nonempty, None]).astype(np.float32)
    return normalize(cent)


def build_centroids_veconly(user_vecs, K, d, N, C_v, B_v, sigma, seed, nonempty):
    """VECTOR-ONLY release: release ONLY the summed vector (1 RDP composition). The count
    channel is not released because the retrieval centroid = normalize(sum_v/count) =
    normalize(sum_v) — the scalar count cancels under L2 normalisation, so counts never
    affect retrieval direction. → tighter eps (9.3 vs 13.9) at identical utility."""
    if sigma == 0.0:
        sv = np.sum(user_vecs, axis=0)
    else:
        sv = secagg_skellam_release([[v] for v in user_vecs], sigma, C_v, B_v, N, seed)[0]
    cent = np.zeros((K, d), dtype=np.float32)
    cent[nonempty] = sv[nonempty].astype(np.float32)
    return normalize(cent)


def build_centroids_joint(user_vecs, user_cnts, K, d, N, sigma, seed, nonempty):
    """JOINT release: concatenate [sum-vector, counts] per user, clip to ONE L2 norm,
    single Skellam release (1 RDP composition → the tighter eps, e.g. 9.3 vs 13.9).
    Takes RAW (unclipped) per-user arrays."""
    combined = [np.concatenate([v.reshape(-1), c]) for v, c in zip(user_vecs, user_cnts)]
    C = float(np.percentile([np.linalg.norm(x) for x in combined], 95))
    combined = [x * min(1.0, C / (np.linalg.norm(x) + 1e-12)) for x in combined]
    B = max(max(float(np.max(np.abs(x))) for x in combined), 1e-6)
    if sigma == 0.0:
        agg = np.sum(combined, axis=0)
    else:
        agg = secagg_skellam_release([[x] for x in combined], sigma, C, B, N, seed)[0]
    sv = agg[:K * d].reshape(K, d)
    sc = np.maximum(agg[K * d:], 1.0)
    cent = np.zeros((K, d), dtype=np.float32)
    cent[nonempty] = (sv[nonempty] / sc[nonempty, None]).astype(np.float32)
    return normalize(cent)


def evidence_recall(q_emb, q_user, note_user, note_ev, note_bucket, centroids, topk):
    """For each query with evidence, fraction of its evidence turns whose bucket is
    in the query's top-k centroids."""
    sims = q_emb @ centroids.T
    topk_buckets = np.argsort(-sims, axis=1)[:, :topk]
    recalls = []
    for qi in range(len(q_emb)):
        u = q_user[qi]
        ev_idx = np.where((note_user == u) & note_ev)[0]
        if len(ev_idx) == 0:
            continue
        ev_buckets = note_bucket[ev_idx]
        hit = np.isin(ev_buckets, topk_buckets[qi])
        recalls.append(float(np.mean(hit)))
    return float(np.mean(recalls)) if recalls else float("nan")


def run(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    eps_points = [float(e) for e in args.eps.split(",")]
    sigma_specs = [("inf (clean)", 0.0)] + [(f"eps={e:g}", single_shot_sigma(e)) for e in eps_points]
    sigma_specs.append(("FL-sigma", args.fl_sigma))

    print("=== Phase-2 LongMemEval real-data utility probe ===")
    _rdp = {"separate": "SEPARATE (2 RDP)", "joint": "JOINT (1 RDP)", "veconly": "VEC-ONLY (1 RDP)"}
    print(f"variant={args.variant} | K={args.K} | proj-d={args.d} | top-{args.topk} | "
          f"release={_rdp[args.release]} | seeds={seeds}")
    print(f"metric = EVIDENCE RECALL (real GT); chance ~= topk/K = {args.topk/args.K:.3f}\n")

    metrics = {lab: [] for lab, _ in sigma_specs}
    stats = {"N": 0, "notes": 0, "queries_with_ev": 0}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        note_raw, note_user, note_ev, q_raw, q_user = get_embeddings(args.variant)
        N = int(note_user.max() + 1)

        # project real 384-d embeddings to tiny-d (fit PCA on notes; apply to queries)
        if args.d < note_raw.shape[1]:
            pca = PCA(n_components=args.d, random_state=seed)
            note_emb = normalize(pca.fit_transform(note_raw)).astype(np.float32)
            q_emb = normalize(pca.transform(q_raw)).astype(np.float32)
        else:
            note_emb = normalize(note_raw).astype(np.float32)
            q_emb = normalize(q_raw).astype(np.float32)

        anchors = normalize(rng.standard_normal((args.K, args.d))).astype(np.float32)
        note_bucket = assign_buckets(note_emb, anchors)

        # per-user (real partition) sum-vector + count
        user_vecs, user_cnts = [], []
        for u in range(N):
            idx = np.where(note_user == u)[0]
            v = np.zeros((args.K, args.d)); c = np.zeros(args.K)
            for i in idx:
                b = note_bucket[i]; v[b] += note_emb[i]; c[b] += 1.0
            user_vecs.append(v); user_cnts.append(c)

        nonempty = np.sum(user_cnts, axis=0) > 0.5  # from RAW counts
        raw_vecs = [v.copy() for v in user_vecs]     # keep raw for the joint path
        raw_cnts = [c.copy() for c in user_cnts]

        C_v = float(np.percentile([np.linalg.norm(v) for v in user_vecs], 95))
        C_c = float(np.percentile([np.linalg.norm(c) for c in user_cnts], 95))
        for u in range(N):
            user_vecs[u] = clip_rows_to_norm(user_vecs[u].reshape(1, -1), C_v).reshape(args.K, args.d)
            user_cnts[u] = clip_rows_to_norm(user_cnts[u].reshape(1, -1), C_c).reshape(args.K)
        B_v = max(float(np.max([np.max(np.abs(v)) for v in user_vecs])), 1e-6)
        B_c = max(float(np.max([np.max(np.abs(c)) for c in user_cnts])), 1e-6)

        stats.update(N=N, notes=len(note_emb),
                     queries_with_ev=int(sum(((note_user == u) & note_ev).any() for u in q_user)))

        for lab, sigma in sigma_specs:
            if args.release == "veconly":
                cent = build_centroids_veconly(user_vecs, args.K, args.d, N,
                                               C_v, B_v, sigma, seed, nonempty)
            elif args.release == "joint":
                cent = build_centroids_joint(raw_vecs, raw_cnts, args.K, args.d, N,
                                             sigma, seed, nonempty)
            else:
                cent = build_centroids(user_vecs, user_cnts, args.K, args.d, N,
                                       C_v, C_c, B_v, B_c, sigma, seed, nonempty)
            metrics[lab].append(evidence_recall(q_emb, q_user, note_user, note_ev,
                                                note_bucket, cent, args.topk))

    print(f"corpus: {stats['N']} users, {stats['notes']} notes, "
          f"{stats['queries_with_ev']} queries w/ evidence\n")
    print(f"{'release':<14}{'sigma':>8}{'evidence-recall@'+str(args.topk):>22}")
    print("-" * 44)
    for lab, sigma in sigma_specs:
        xs = metrics[lab]
        m = float(np.mean(xs)); s = float(np.std(xs)) if len(xs) > 1 else 0.0
        print(f"{lab:<14}{sigma:>8.3f}{m:>16.3f} +/-{s:<5.2f}")
    print(f"\nchance ~= {args.topk/args.K:.3f}. Does clean >> chance (mapping works) and DP hold up?")

    if args.json:
        rows = [{"label": lab, "sigma": sigma,
                 "mean": float(np.mean(metrics[lab])),
                 "std": float(np.std(metrics[lab])) if len(metrics[lab]) > 1 else 0.0,
                 "per_seed": [float(x) for x in metrics[lab]]}
                for lab, sigma in sigma_specs]
        out = {"script": "longmemeval_probe", "metric": f"evidence-recall@{args.topk}",
               "chance": args.topk / args.K, "seeds": seeds,
               "config": {"variant": args.variant, "K": args.K, "d": args.d,
                          "topk": args.topk, "release": args.release, "eps": args.eps},
               "stats": stats, "rows": rows}
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.json, "w"), indent=2)
        print(f"[json] wrote {args.json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", type=str, default="oracle", choices=["oracle", "s"])
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--release", type=str, default="separate",
                   choices=["separate", "joint", "veconly"],
                   help="separate=2 releases; joint=1 concat release; veconly=1 vector-only release")
    p.add_argument("--eps", type=str, default="16,8,3")
    p.add_argument("--fl_sigma", type=float, default=2.854)
    p.add_argument("--seeds", type=str, default="0,1")
    p.add_argument("--json", type=str, default=None, help="optional path to dump aggregated metrics")
    run(p.parse_args())