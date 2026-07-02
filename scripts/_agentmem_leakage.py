"""
Phase-4 leakage-drop endpoint (D5) for the agent-memory bet — de-risking spike.

QUESTION
--------
Phase 0 showed the DP centroid memory stays USEFUL (retrieval survives at eps=8).
The PAPER's headline needs the other half: does the DP noise that is "nearly free"
for utility BUY a measurable drop in privacy-attack success? i.e. a POSITIVE,
measurable endpoint (attack-success down).

This is a spike, not the full MEXTRA/MRMMIA LLM attacks (those need real notes + an
LLM, Phase 2/3). It implements the MEASUREMENT PRINCIPLE those attacks embody,
against the released centroid memory:

  (1) MEMBERSHIP INFERENCE (MRMMIA analog) — the standard threshold attack.
      Adversary holds a candidate note embedding x, assigns it to its shared bucket k,
      scores s(x) = cos(x, centroid[k]). Members contributed to (pulled) their own
      centroid, so score higher than same-distribution non-members. Metric = ROC-AUC
      (0.5 = no leakage / perfect privacy). DP noise on the centroid should push AUC
      toward 0.5.

  (2) EXTRACTION / RECONSTRUCTION gap (MEXTRA analog).
      A centroid is a lossy reconstruction of its members. Measure how much better the
      centroid reconstructs a true MEMBER note (max cosine to in-bucket members) than a
      same-distribution NON-MEMBER. gap = recon_member - recon_nonmember; DP should
      shrink it.

Honest MIA setup: members and non-members are BOTH drawn from the training distribution
(train pool is split into aggregated-members vs held-out non-members) — no train/test
distribution confound. Utility (topic-acc) reported alongside so the eps-sweep is a
privacy-UTILITY tradeoff curve (the paper's money figure).

Faithful crypto path: reuses scripts/_agentmem_probe.py helpers verbatim (real
privacy_utils quantize/skellam/dequantize). Multi-seed + mean+/-std.

Usage:
  .venv/bin/python scripts/_agentmem_leakage.py --N 100 --K 32 --d 32 --seeds 0,1
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _agentmem_probe import (  # noqa: E402
    build_embeddings,
    dirichlet_partition,
    assign_buckets,
    clip_rows_to_norm,
    secagg_skellam_release,
    single_shot_sigma,
    topic_acc,
)


def build_dp_centroids(user_vecs, user_cnts, K, d, N, C_v, C_c, B_v, B_c, sigma, seed, nonempty):
    per_user_v = [[user_vecs[u]] for u in range(N)]
    per_user_c = [[user_cnts[u].reshape(K, 1)] for u in range(N)]
    if sigma == 0.0:
        sum_v = np.sum(user_vecs, axis=0)
        sum_c = np.sum(user_cnts, axis=0)
    else:
        sum_v = secagg_skellam_release(per_user_v, sigma, C_v, B_v, N, seed)[0]
        sum_c = secagg_skellam_release(per_user_c, sigma, C_c, B_c, N, seed)[0].reshape(K)
    sum_c = np.maximum(sum_c, 1.0)
    cent = np.zeros((K, d), dtype=np.float32)
    cent[nonempty] = (sum_v[nonempty] / sum_c[nonempty, None]).astype(np.float32)
    return normalize(cent)


def mia_auc(emb, bucket, centroids):
    """score = cos(x, assigned centroid). emb rows already normalized."""
    scores = np.sum(emb * centroids[bucket], axis=1)
    return scores


def extraction_gap(members, mem_bucket, nonmembers, non_bucket, centroids, K):
    """For each bucket: max cos(centroid, in-bucket member) vs (centroid, non-member)."""
    rm, rn = [], []
    for k in range(K):
        mk = members[mem_bucket == k]
        nk = nonmembers[non_bucket == k]
        if len(mk) and np.linalg.norm(centroids[k]) > 0:
            rm.append(float(np.max(mk @ centroids[k])))
        if len(nk) and np.linalg.norm(centroids[k]) > 0:
            rn.append(float(np.max(nk @ centroids[k])))
    return (float(np.mean(rm)) if rm else 0.0), (float(np.mean(rn)) if rn else 0.0)


def run(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    eps_points = [float(e) for e in args.eps.split(",")]
    sigma_specs = [("inf (clean)", 0.0)]
    for e in eps_points:
        sigma_specs.append((f"eps={e:g}", single_shot_sigma(e)))
    sigma_specs.append(("FL-sigma", args.fl_sigma))

    print("=== Phase-4 agent-memory LEAKAGE-DROP spike ===")
    print(f"N={args.N} | K={args.K} | d={args.d} | members=non-members={args.M} (same dist) | seeds={seeds}")
    print("MIA-AUC: 0.500 = NO leakage (ideal privacy). extraction gap: lower = less leakage.\n")

    metrics = {lab: {"auc": [], "lc_auc": [], "gap": [], "topic": []} for lab, _ in sigma_specs}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        Etr, ytr, Ete, yte, _ = build_embeddings(args.d, seed, args.embedder)
        n_topics = ytr.max() + 1
        anchors = normalize(rng.standard_normal((args.K, args.d))).astype(np.float32)

        # split TRAIN into aggregated pool (members) vs held-out non-members (same dist)
        perm = rng.permutation(len(Etr))
        n_pool = int(0.8 * len(Etr))
        pool_idx, hold_idx = perm[:n_pool], perm[n_pool:]

        # queries for utility (topic-acc), from the test split
        qsel = rng.choice(len(Ete), size=min(args.M, len(Ete)), replace=False)
        q_emb, q_lab = Ete[qsel], yte[qsel]

        bucket_of = assign_buckets(Etr, anchors)

        # partition ONLY the pool across users (non-IID by topic)
        pool_labels = ytr[pool_idx]
        local_parts = dirichlet_partition(pool_labels, args.N, args.alpha, rng)
        user_idx = [[pool_idx[j] for j in part] for part in local_parts]

        user_vecs, user_cnts = [], []
        for u in range(args.N):
            v = np.zeros((args.K, args.d)); c = np.zeros(args.K)
            for idx in user_idx[u]:
                b = bucket_of[idx]; v[b] += Etr[idx]; c[b] += 1.0
            user_vecs.append(v); user_cnts.append(c)

        C_v = float(np.percentile([np.linalg.norm(v) for v in user_vecs], 95))
        C_c = float(np.percentile([np.linalg.norm(c) for c in user_cnts], 95))
        for u in range(args.N):
            user_vecs[u] = clip_rows_to_norm(user_vecs[u].reshape(1, -1), C_v).reshape(args.K, args.d)
            user_cnts[u] = clip_rows_to_norm(user_cnts[u].reshape(1, -1), C_c).reshape(args.K)
        B_v = max(float(np.max([np.max(np.abs(v)) for v in user_vecs])), 1e-6)
        B_c = max(float(np.max([np.max(np.abs(c)) for c in user_cnts])), 1e-6)

        sum_c_clean = np.sum(user_cnts, axis=0)
        nonempty = sum_c_clean > 0.5

        # centroid topic labels (data-independent buckets)
        cent_topic = np.zeros(args.K, dtype=int)
        for k in range(args.K):
            docs_k = np.where(bucket_of == k)[0]
            if len(docs_k):
                cent_topic[k] = np.bincount(ytr[docs_k], minlength=n_topics).argmax()

        # balanced member / non-member sets (same distribution)
        m_sel = rng.choice(pool_idx, size=min(args.M, len(pool_idx)), replace=False)
        n_sel = rng.choice(hold_idx, size=min(args.M, len(hold_idx)), replace=False)
        mem_emb, mem_b = Etr[m_sel], bucket_of[m_sel]
        non_emb, non_b = Etr[n_sel], bucket_of[n_sel]
        y_true = np.concatenate([np.ones(len(m_sel)), np.zeros(len(n_sel))])

        # low-count tail: members/non-members whose bucket has <= T contributors
        # (where a SUM mechanism leaks most; DP should protect exactly these)
        T = args.lowcount
        mem_cnt = sum_c_clean[mem_b]
        non_cnt = sum_c_clean[non_b]
        lc_m = mem_cnt <= T
        lc_n = non_cnt <= T
        lc_frac = float(np.mean(lc_m))

        for lab, sigma in sigma_specs:
            cent = build_dp_centroids(user_vecs, user_cnts, args.K, args.d, args.N,
                                      C_v, C_c, B_v, B_c, sigma, seed, nonempty)
            s_mem = mia_auc(mem_emb, mem_b, cent)
            s_non = mia_auc(non_emb, non_b, cent)
            auc = roc_auc_score(y_true, np.concatenate([s_mem, s_non]))
            # stratified AUC on the low-count tail
            if lc_m.sum() >= 10 and lc_n.sum() >= 10:
                y_lc = np.concatenate([np.ones(lc_m.sum()), np.zeros(lc_n.sum())])
                lc_auc = roc_auc_score(y_lc, np.concatenate([s_mem[lc_m], s_non[lc_n]]))
            else:
                lc_auc = float("nan")
            rm, rn = extraction_gap(mem_emb, mem_b, non_emb, non_b, cent, args.K)
            keep = nonempty
            tacc = topic_acc(q_emb, q_lab, cent[keep], cent_topic[keep])
            metrics[lab]["auc"].append(float(auc))
            metrics[lab]["lc_auc"].append(float(lc_auc))
            metrics[lab]["gap"].append(rm - rn)
            metrics[lab]["topic"].append(tacc)
        metrics["_lc_frac"] = metrics.get("_lc_frac", [])
        metrics["_lc_frac"].append(lc_frac)

    def ms(xs):
        return float(np.mean(xs)), (float(np.std(xs)) if len(xs) > 1 else 0.0)

    lc_frac = float(np.mean(metrics.get("_lc_frac", [0.0])))
    print(f"low-count tail: buckets with <= {args.lowcount} contributors "
          f"({lc_frac*100:.0f}% of members)\n")
    print(f"{'release':<14}{'sigma':>7}{'MIA-AUC':>15}{'AUC(low-cnt)':>16}{'extract-gap':>15}{'utility':>14}")
    print("-" * 81)
    for lab, sigma in sigma_specs:
        a_m, a_s = ms(metrics[lab]["auc"])
        l_m, l_s = ms(metrics[lab]["lc_auc"])
        g_m, g_s = ms(metrics[lab]["gap"])
        t_m, t_s = ms(metrics[lab]["topic"])
        print(f"{lab:<14}{sigma:>7.3f}{a_m:>9.3f}+/-{a_s:<4.2f}{l_m:>10.3f}+/-{l_s:<4.2f}"
              f"{g_m:>9.3f}+/-{g_s:<4.2f}{t_m:>8.3f}+/-{t_s:<4.2f}")
    clean_auc = ms(metrics["inf (clean)"]["auc"])[0]
    clean_lc = ms(metrics["inf (clean)"]["lc_auc"])[0]
    print(f"\nclean AUC(all)={clean_auc:.3f}, clean AUC(low-count tail)={clean_lc:.3f}; ideal=0.500.")
    print("Headline: does AUC fall toward 0.5 as eps tightens (esp. the leaky tail), while utility holds?")

    if args.json:
        import json as _json
        rows = []
        for lab, sigma in sigma_specs:
            a_m, a_s = ms(metrics[lab]["auc"])
            l_m, l_s = ms(metrics[lab]["lc_auc"])
            g_m, g_s = ms(metrics[lab]["gap"])
            t_m, t_s = ms(metrics[lab]["topic"])
            rows.append({"label": lab, "sigma": sigma,
                         "auc_mean": a_m, "auc_std": a_s,
                         "lc_auc_mean": l_m, "lc_auc_std": l_s,
                         "gap_mean": g_m, "gap_std": g_s,
                         "topic_mean": t_m, "topic_std": t_s,
                         "auc_per_seed": [float(x) for x in metrics[lab]["auc"]],
                         "lc_auc_per_seed": [float(x) for x in metrics[lab]["lc_auc"]]})
        out = {"script": "agentmem_leakage", "metric": "MIA-AUC / tail-AUC / extract-gap / utility",
               "seeds": seeds, "embedder": args.embedder, "lc_frac": lc_frac,
               "config": {"N": args.N, "K": args.K, "d": args.d, "M": args.M,
                          "alpha": args.alpha, "lowcount": args.lowcount, "eps": args.eps},
               "rows": rows}
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        _json.dump(out, open(args.json, "w"), indent=2)
        print(f"[json] wrote {args.json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=100)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--M", type=int, default=2000)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--eps", type=str, default="16,8,3,1")
    p.add_argument("--fl_sigma", type=float, default=2.854)
    p.add_argument("--lowcount", type=int, default=3)
    p.add_argument("--embedder", type=str, default="tfidf", choices=["tfidf", "st"])
    p.add_argument("--seeds", type=str, default="0,1")
    p.add_argument("--json", type=str, default=None, help="optional path to dump aggregated metrics")
    run(p.parse_args())