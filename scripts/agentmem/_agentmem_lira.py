"""
Calibrated (LiRA-style) membership inference + a reconstruction/decode extraction attack
against the SecAgg+Skellam centroid memory. This UPGRADES the raw cosine-AUC proxy in
`_agentmem_leakage.py` / `_longmemeval_leakage.py` to the modern MIA standard, staying inside
the paper's threat model (the adversary observes the *released* pool, not a live agent).

Why a real LiRA is feasible here (and is a genuine strength): unlike model-training MIA, our
"aggregation" is cheap numpy, so we can run many SHADOW releases and estimate, per target note,
the IN-distribution (note aggregated into the pool) and OUT-distribution (note held out) of its
membership score. This is offline LiRA (Carlini et al., "Membership Inference Attacks From First
Principles", S&P 2022): fit per-target Gaussians to the IN/OUT shadow scores and use the
likelihood ratio as the test statistic. We report ROC-AUC AND the low-FPR TPR (TPR@1%/0.1%FPR),
which is where a naive AUC-only proxy hides real leakage.

Two attacks, both over an eps sweep and stratified by the low-count tail:
  (1) LiRA MIA  — statistic = logN(s|mu_in,sig_in) - logN(s|mu_out,sig_out), s = cos(x, centroid).
  (2) RECON/DECODE extraction (MEXTRA analog) — nearest-note decode of each released centroid;
      attack success = top-1 decoded note is a true in-bucket member. DP should drive both to chance.

Payloads: real LongMemEval oracle embeddings (default) or the ST/TF-IDF controls.

Usage:
  .venv/Scripts/python.exe scripts/agentmem/_agentmem_lira.py --source oracle --K 1024 --d 32 \
       --shadows 64 --targets 1500 --seeds 0,1,2 --eps 16,8,3,1 --json out.json
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
from _agentmem_probe import (  # noqa: E402
    assign_buckets, clip_rows_to_norm, secagg_skellam_release, single_shot_sigma,
)

_CACHE = Path(__file__).resolve().parents[2] / "experiment_results"


# ------------------------------- data -------------------------------
def load_notes(source, d, seed):
    """Return (note_emb[N,d] L2-normalized, ) for the note universe."""
    if source in ("oracle", "s"):
        from _longmemeval_data import get_embeddings
        raw, _, _, _, _ = get_embeddings(source)
    elif source == "st":
        from _agentmem_probe import build_embeddings
        Etr, _, Ete, _, _ = build_embeddings(384, seed, "st")  # 384-d raw ST, reduce below
        raw = np.concatenate([Etr, Ete], axis=0)
    else:
        raise ValueError(source)
    raw = raw.astype(np.float32)
    if d < raw.shape[1]:
        raw = PCA(n_components=d, random_state=seed).fit_transform(raw)
    return normalize(raw).astype(np.float32)


# --------------------------- one DP release ---------------------------
def release_centroids(note_emb, in_mask, user_of, K, d, N, anchors, sigma, seed):
    """Vector-only SecAgg+Skellam release of the pooled centroid memory over the IN notes.
    Returns (centroids[K,d] normalized, contributor_count[K])."""
    bucket = assign_buckets(note_emb, anchors)
    uv = [np.zeros((K, d), dtype=np.float32) for _ in range(N)]
    cnt = np.zeros(K)
    idx_in = np.where(in_mask)[0]
    for i in idx_in:
        u = user_of[i]; b = bucket[i]
        uv[u][b] += note_emb[i]; cnt[b] += 1.0
    C_v = float(np.percentile([np.linalg.norm(v) for v in uv], 95)) or 1.0
    for u in range(N):
        uv[u] = clip_rows_to_norm(uv[u].reshape(1, -1), C_v).reshape(K, d)
    B_v = max(float(np.max([np.max(np.abs(v)) for v in uv])), 1e-6)
    if sigma == 0.0:
        sv = np.sum(uv, axis=0)
    else:
        sv = secagg_skellam_release([[v] for v in uv], sigma, C_v, B_v, N, seed)[0]
    cent = normalize(sv.astype(np.float32))
    return cent, cnt, bucket


def _score(note_emb, targets, bucket, cent):
    """Membership score s(x) = cos(x, centroid[bucket(x)]) for the target rows."""
    return np.sum(note_emb[targets] * cent[bucket[targets]], axis=1)


# ------------------------------- LiRA -------------------------------
def tpr_at_fpr(y, score, fpr_target):
    """TPR at a fixed low FPR (threshold set on the negatives)."""
    neg = score[y == 0]; pos = score[y == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    thr = np.quantile(neg, 1.0 - fpr_target)
    return float(np.mean(pos > thr))


def run_lira(note_emb, K, d, N, sigma, seed, n_targets, n_shadows, lowcount):
    """Offline LiRA at one sigma. Returns dict of AUC / TPR@1% / TPR@0.1% for all + low-count tail."""
    rng = np.random.default_rng(seed)
    n = len(note_emb)
    anchors = normalize(rng.standard_normal((K, d))).astype(np.float32)
    user_of = rng.integers(0, N, size=n)  # fixed note->user map across shadows

    targets = rng.choice(n, size=min(n_targets, n), replace=False)
    others = np.setdiff1d(np.arange(n), targets)  # always-in background
    T = len(targets)

    in_scores = [[] for _ in range(T)]   # per-target shadow scores when target is IN
    out_scores = [[] for _ in range(T)]
    cnt_accum = np.zeros(K)              # avg contributor count per bucket (for tail strat.)
    for s in range(n_shadows):
        incl = rng.random(T) < 0.5                       # random half of targets IN this shadow
        in_mask = np.zeros(n, dtype=bool)
        in_mask[others] = True
        in_mask[targets[incl]] = True
        cent, cnt, bucket = release_centroids(note_emb, in_mask, user_of, K, d, N,
                                              anchors, sigma, seed * 10_000 + s)
        cnt_accum += cnt
        sc = _score(note_emb, targets, bucket, cent)
        for t in range(T):
            (in_scores if incl[t] else out_scores)[t].append(sc[t])
    cnt_accum /= n_shadows

    # per-target Gaussian fits on a FIT half of the shadows; score the disjoint EVAL half
    # (avoids the train-on-test bias that would optimistically inflate AUC under DP).
    stat, label, tgt_cnt = [], [], []
    bucket_full = assign_buckets(note_emb, anchors)
    for t in range(T):
        ins = np.array(in_scores[t]); outs = np.array(out_scores[t])
        if len(ins) < 4 or len(outs) < 4:
            continue
        hi, ho = len(ins) // 2, len(outs) // 2
        fit_i, ev_i = ins[:hi], ins[hi:]          # disjoint fit / eval splits
        fit_o, ev_o = outs[:ho], outs[ho:]
        mu_i, sd_i = fit_i.mean(), fit_i.std() + 1e-6
        mu_o, sd_o = fit_o.mean(), fit_o.std() + 1e-6
        def llr(x):  # log-likelihood ratio, in vs out (higher => member)
            return (-0.5 * ((x - mu_i) / sd_i) ** 2 - np.log(sd_i)) \
                 - (-0.5 * ((x - mu_o) / sd_o) ** 2 - np.log(sd_o))
        cb = cnt_accum[bucket_full[targets[t]]]
        for x in ev_i:
            stat.append(llr(x)); label.append(1); tgt_cnt.append(cb)
        for x in ev_o:
            stat.append(llr(x)); label.append(0); tgt_cnt.append(cb)
    stat = np.array(stat); label = np.array(label); tgt_cnt = np.array(tgt_cnt)

    def metrics(mask):
        y, sc = label[mask], stat[mask]
        if y.sum() < 5 or (1 - y).sum() < 5:
            return float("nan"), float("nan"), float("nan")
        return (roc_auc_score(y, sc), tpr_at_fpr(y, sc, 0.01), tpr_at_fpr(y, sc, 0.001))
    auc, t1, t01 = metrics(np.ones(len(label), dtype=bool))
    lc = tgt_cnt <= lowcount
    lc_auc, lc_t1, _ = metrics(lc)
    return {"auc": auc, "tpr1": t1, "tpr01": t01, "lc_auc": lc_auc, "lc_tpr1": lc_t1,
            "lc_frac": float(np.mean(lc))}


# --------------------- reconstruction / decode ---------------------
def run_extraction(note_emb, K, d, N, sigma, seed, n_pool):
    """MEXTRA analog: decode each released centroid to its nearest note; success = the
    top-1 decoded note is a true in-bucket member. Reports member vs held-out decode acc."""
    rng = np.random.default_rng(seed + 777)
    n = len(note_emb)
    anchors = normalize(rng.standard_normal((K, d))).astype(np.float32)
    user_of = rng.integers(0, N, size=n)
    perm = rng.permutation(n)
    pool = perm[:min(n_pool, int(0.8 * n))]
    in_mask = np.zeros(n, dtype=bool); in_mask[pool] = True
    cent, cnt, bucket = release_centroids(note_emb, in_mask, user_of, K, d, N, anchors, sigma, seed)

    sims = note_emb @ cent.T          # [n, K] cosine of every note to every centroid
    top1 = np.argmax(sims, axis=0)    # nearest note to each centroid
    hits, tail_hits, tail_tot = 0, 0, 0
    for k in range(K):
        if cnt[k] < 0.5:
            continue
        nn = top1[k]
        is_member = in_mask[nn] and bucket[nn] == k
        hits += int(is_member)
        if cnt[k] <= 3:
            tail_tot += 1; tail_hits += int(is_member)
    nk = int(np.sum(cnt >= 0.5))
    return {"decode_acc": hits / max(nk, 1),
            "tail_decode_acc": (tail_hits / tail_tot) if tail_tot else float("nan"),
            "chance": 1.0 / n}


def run(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    eps_points = [float(e) for e in args.eps.split(",")]
    specs = [("inf (clean)", 0.0)] + [(f"eps={e:g}", single_shot_sigma(e)) for e in eps_points]
    specs.append(("FL-sigma", args.fl_sigma))

    note_emb = load_notes(args.source, args.d, seeds[0])
    N = args.N
    print(f"=== Calibrated LiRA MIA + recon-decode extraction ===")
    print(f"source={args.source} notes={len(note_emb)} | K={args.K} d={args.d} N={N} "
          f"| shadows={args.shadows} targets={args.targets} | seeds={seeds}")
    print("AUC/TPR: higher = more leakage. DP should drive toward chance (AUC 0.5, TPR~FPR).\n")

    agg = {lab: {k: [] for k in ["auc", "tpr1", "tpr01", "lc_auc", "lc_tpr1",
                                  "decode_acc", "tail_decode_acc"]} for lab, _ in specs}
    lc_frac = []
    for seed in seeds:
        for lab, sigma in specs:
            m = run_lira(note_emb, args.K, args.d, N, sigma, seed,
                         args.targets, args.shadows, args.lowcount)
            e = run_extraction(note_emb, args.K, args.d, N, sigma, seed, args.pool)
            for k in ["auc", "tpr1", "tpr01", "lc_auc", "lc_tpr1"]:
                agg[lab][k].append(m[k])
            agg[lab]["decode_acc"].append(e["decode_acc"])
            agg[lab]["tail_decode_acc"].append(e["tail_decode_acc"])
            if lab == "inf (clean)":
                lc_frac.append(m["lc_frac"])

    def ms(xs):
        xs = [x for x in xs if x == x]  # drop NaN
        return (float(np.mean(xs)) if xs else float("nan"),
                float(np.std(xs)) if len(xs) > 1 else 0.0)

    print(f"low-count tail (<= {args.lowcount} contributors): {np.mean(lc_frac)*100:.1f}% of targets\n")
    print(f"{'release':<13}{'AUC':>8}{'TPR@1%':>9}{'TPR@.1%':>9}{'tailAUC':>9}"
          f"{'tailTPR1':>9}{'decode':>8}{'tailDec':>8}")
    print("-" * 73)
    rows = []
    for lab, sigma in specs:
        r = {k: ms(agg[lab][k])[0] for k in agg[lab]}
        rows.append({"label": lab, "sigma": sigma, **r})
        def g(k): return r[k] if r[k] == r[k] else float("nan")
        print(f"{lab:<13}{g('auc'):>8.3f}{g('tpr1'):>9.3f}{g('tpr01'):>9.3f}{g('lc_auc'):>9.3f}"
              f"{g('lc_tpr1'):>9.3f}{g('decode_acc'):>8.3f}{g('tail_decode_acc'):>8.3f}")
    print("\nAUC 0.5 / TPR@1%~0.01 / decode~chance => attack defeated. Clean should be >> that.")

    if args.json:
        out = {"script": "agentmem_lira", "metric": "LiRA-AUC/TPR + recon-decode",
               "source": args.source, "seeds": seeds, "lc_frac": float(np.mean(lc_frac)),
               "config": {"N": N, "K": args.K, "d": args.d, "shadows": args.shadows,
                          "targets": args.targets, "pool": args.pool, "lowcount": args.lowcount,
                          "eps": args.eps},
               "rows": rows}
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.json, "w", encoding="utf-8"), indent=2)
        print(f"[json] wrote {args.json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="oracle", choices=["oracle", "s", "st"])
    p.add_argument("--N", type=int, default=500)
    p.add_argument("--K", type=int, default=1024)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--shadows", type=int, default=64)
    p.add_argument("--targets", type=int, default=1500)
    p.add_argument("--pool", type=int, default=8000, help="notes aggregated for the extraction release")
    p.add_argument("--lowcount", type=int, default=3)
    p.add_argument("--eps", type=str, default="16,8,3,1")
    p.add_argument("--fl_sigma", type=float, default=2.854)
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--json", type=str, default=None)
    run(p.parse_args())
