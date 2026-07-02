"""
Phase-0 go/no-go probe for the D1+D5 bet: private federated AGENT-MEMORY aggregation.

QUESTION THIS ANSWERS
---------------------
SecAgg+Skellam is a *summation* primitive: the server only ever learns Sum_i x_i.
So "aggregate agent memories across users" only works if the shared memory is a
SUMMABLE aggregate. The design under test = DP CENTROID MEMORY:
  - fix K data-independent buckets (random unit anchors, fixed seed);
  - each user contributes, per bucket k: (v_ik = sum of their note-embeddings in k,
    c_ik = count in k);
  - SecAgg-sum across users -> per-bucket (sum-vector, count);
  - DP centroid[k] = noised_sum_v[k] / max(noised_count[k], 1).
The shared "memory pool" = K DP centroids used for retrieval.

GO/NO-GO METRIC
---------------
Does the DP centroid memory still RETRIEVE like the noise-free aggregate?
  (1) recall@k of DP centroids vs noise-free centroids (no labels needed);
  (2) topic-retrieval accuracy: does a held-out query still land on a centroid
      whose dominant 20-newsgroups topic matches the query's topic.

FAITHFULNESS
------------
The privacy path is byte-identical to the FL code: it imports the PROJECT's real
quantize / apply_distributed_skellam_noise / dequantize from privacy_utils. Only the
PAYLOAD changes (memory-note embeddings instead of gradient deltas). The zero-sum
SecAgg masks are omitted because they cancel EXACTLY under summation and do not
affect the released (noised) sum — the DP-relevant part is the Skellam noise, which
is applied verbatim.

Embedder: TF-IDF -> TruncatedSVD dense vectors (fully offline). The mechanism's
effect on a summed centroid is embedder-agnostic to first order; Phase 2 swaps in
real A-MEM / Mem0 sentence embeddings.

Usage:
  .venv/bin/python scripts/_agentmem_probe.py --N 50 --K 64 --d 256 --seeds 0,1
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qpriviot_fl.privacy_utils import (  # noqa: E402
    quantize,
    dequantize,
    apply_distributed_skellam_noise,
)

RANGE_MAX = 1_000_000
DELTA = 1e-5


def single_shot_sigma(eps: float) -> float:
    """Gaussian-mechanism noise multiplier for a SINGLE (eps, delta) release."""
    return math.sqrt(2.0 * math.log(1.25 / DELTA)) / eps


def secagg_skellam_release(per_user_arrays, sigma, clip_norm, quant_bound, num_clients, seed):
    """Run the PROJECT's real crypto path on a list-of-arrays payload, per user,
    and return the (dequantized) DP-noised sum across users.

    per_user_arrays: list (len=num_clients) of [array_a, array_b, ...] (same shapes).
    Masks omitted (cancel exactly under summation); Skellam noise applied verbatim.
    """
    n_arrays = len(per_user_arrays[0])
    agg_int = None
    for i, arrays in enumerate(per_user_arrays):
        q = quantize(arrays, clip_range=quant_bound, range_max=RANGE_MAX)
        if sigma > 0:
            q = apply_distributed_skellam_noise(
                q,
                sigma=sigma,
                clip_norm=clip_norm,
                num_clients=num_clients,
                quantization_bound=quant_bound,
                range_max=RANGE_MAX,
                distributed=True,           # SecAgg: per-client var = (sigma*C*s)^2 / N
                seed=(seed * 1000003 + i * 31 + 17) & 0x7FFFFFFF,
            )
        if agg_int is None:
            agg_int = [a.astype(np.int64).copy() for a in q]
        else:
            for j in range(n_arrays):
                agg_int[j] += q[j].astype(np.int64)
    return dequantize(agg_int, clip_range=quant_bound, range_max=RANGE_MAX)


def clip_rows_to_norm(mat, C):
    """Clip each row of `mat` to L2 norm C (client-level DP sensitivity bound)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    factor = np.minimum(1.0, C / (norms + 1e-12))
    return mat * factor


_ST_CACHE = str(Path(__file__).resolve().parents[1] / "experiment_results" / "_st_20ng_raw.npz")


def _encode_st_raw(model_name="all-MiniLM-L6-v2"):
    """Encode all 20newsgroups docs once with a REAL sentence encoder; cache to disk.
    Returns raw (uncompressed) embeddings + labels for train/test."""
    if Path(_ST_CACHE).exists():
        z = np.load(_ST_CACHE)
        return z["Etr"], z["ytr"], z["Ete"], z["yte"], list(z["names"])
    from sentence_transformers import SentenceTransformer
    tr = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    te = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
    m = SentenceTransformer(model_name)
    Etr = m.encode(tr.data, batch_size=128, show_progress_bar=True,
                   normalize_embeddings=False).astype(np.float32)
    Ete = m.encode(te.data, batch_size=128, show_progress_bar=True,
                   normalize_embeddings=False).astype(np.float32)
    Path(_ST_CACHE).parent.mkdir(parents=True, exist_ok=True)
    np.savez(_ST_CACHE, Etr=Etr, ytr=np.array(tr.target), Ete=Ete,
             yte=np.array(te.target), names=np.array(tr.target_names))
    return Etr, np.array(tr.target), Ete, np.array(te.target), list(tr.target_names)


def build_embeddings(d, seed, embedder="tfidf"):
    """Return (Etr, ytr, Ete, yte, topic_names), all embeddings L2-normalized to dim d.

    embedder='tfidf' : TF-IDF -> TruncatedSVD(d)  (offline, original Phase-0 path)
    embedder='st'    : REAL all-MiniLM-L6-v2 (384-d) -> PCA(d) projection if d<384.
                       Tests whether the findings survive real embedding geometry, and
                       makes the projection dim d an explicit tiny-d design lever.
    """
    if embedder == "tfidf":
        train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
        test = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
        tfidf = TfidfVectorizer(max_features=20000, stop_words="english", min_df=3)
        Xtr = tfidf.fit_transform(train.data)
        Xte = tfidf.transform(test.data)
        svd = TruncatedSVD(n_components=d, random_state=seed)
        Etr = normalize(svd.fit_transform(Xtr)).astype(np.float32)
        Ete = normalize(svd.transform(Xte)).astype(np.float32)
        return Etr, np.array(train.target), Ete, np.array(test.target), train.target_names

    if embedder == "st":
        rtr, ytr, rte, yte, names = _encode_st_raw()
        if d < rtr.shape[1]:
            pca = PCA(n_components=d, random_state=seed)
            Etr = pca.fit_transform(rtr)
            Ete = pca.transform(rte)
        else:
            Etr, Ete = rtr, rte
        return (normalize(Etr).astype(np.float32), ytr,
                normalize(Ete).astype(np.float32), yte, names)

    raise ValueError(f"unknown embedder {embedder}")


def dirichlet_partition(labels, num_users, alpha, rng):
    """Non-IID by-topic partition of doc indices across users (mirrors the FL setting)."""
    n_classes = labels.max() + 1
    user_idx = [[] for _ in range(num_users)]
    for c in range(n_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        props = rng.dirichlet([alpha] * num_users)
        cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
        for u, chunk in enumerate(np.split(idx_c, cuts)):
            user_idx[u].extend(chunk.tolist())
    return user_idx


def assign_buckets(emb, anchors):
    """Nearest fixed anchor by cosine (== argmax dot for normalized vectors)."""
    return np.argmax(emb @ anchors.T, axis=1)


def recall_at_k(q_emb, cent_a, cent_b, k):
    """Mean overlap@k between top-k centroids under memory A vs memory B, per query."""
    sim_a = q_emb @ cent_a.T
    sim_b = q_emb @ cent_b.T
    top_a = np.argsort(-sim_a, axis=1)[:, :k]
    top_b = np.argsort(-sim_b, axis=1)[:, :k]
    overlaps = [len(set(top_a[i]) & set(top_b[i])) / k for i in range(len(q_emb))]
    return float(np.mean(overlaps))


def topic_acc(q_emb, q_labels, centroids, cent_topic):
    """Nearest-centroid topic prediction accuracy."""
    pred_cent = np.argmax(q_emb @ centroids.T, axis=1)
    pred_topic = cent_topic[pred_cent]
    return float(np.mean(pred_topic == q_labels))


def run(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    eps_points = [float(e) for e in args.eps.split(",")]

    # sigma sweep: noise-free, then one per eps (single-shot), plus FL-composed stress point.
    sigma_specs = [("inf", 0.0)]
    for e in eps_points:
        sigma_specs.append((f"eps={e:g}", single_shot_sigma(e)))
    sigma_specs.append(("FL-sigma(T=20)", args.fl_sigma))

    print(f"=== Phase-0 agent-memory DP probe ===")
    print(f"N={args.N} users | K={args.K} buckets | d={args.d} | queries={args.M} | seeds={seeds}")
    print(f"payload=20newsgroups TF-IDF->SVD (offline) | crypto path=privacy_utils (verbatim)")
    print(f"delta={DELTA:g} | single-shot sigma(eps): "
          + ", ".join(f"{e:g}->{single_shot_sigma(e):.3f}" for e in eps_points))
    print(f"NOTE: two Skellam releases (sum-vector + counts); reported eps is PER-release,")
    print(f"      total user-level eps ~2x under basic composition (conservative).\n")

    # collect metrics[label] = {"recall": [...], "topic_dp": [...]}
    metrics = {lab: {"recall": [], "topic_dp": []} for lab, _ in sigma_specs}
    topic_clean_all, topic_rawknn_all = [], []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        Etr, ytr, Ete, yte, topic_names = build_embeddings(args.d, seed, args.embedder)
        n_topics = ytr.max() + 1

        # data-independent bucket anchors (fixed given seed; shared across all users)
        anchors = normalize(rng.standard_normal((args.K, args.d))).astype(np.float32)

        # held-out queries
        qsel = rng.choice(len(Ete), size=min(args.M, len(Ete)), replace=False)
        q_emb, q_lab = Ete[qsel], yte[qsel]

        # partition memory notes across users (non-IID by topic)
        user_idx = dirichlet_partition(ytr, args.N, args.alpha, rng)
        bucket_of = assign_buckets(Etr, anchors)

        # per-user contributions: sum-vector (K,d) and counts (K,)
        user_vecs, user_cnts = [], []
        for u in range(args.N):
            v = np.zeros((args.K, args.d), dtype=np.float64)
            c = np.zeros(args.K, dtype=np.float64)
            for idx in user_idx[u]:
                b = bucket_of[idx]
                v[b] += Etr[idx]
                c[b] += 1.0
            user_vecs.append(v)
            user_cnts.append(c)

        # client-level DP sensitivity: clip each user's stacked contribution.
        # C_v = 95th pct of per-user ||flattened sum-vector||; C_c likewise for counts.
        v_norms = np.array([np.linalg.norm(v) for v in user_vecs])
        c_norms = np.array([np.linalg.norm(c) for c in user_cnts])
        C_v = float(np.percentile(v_norms, 95))
        C_c = float(np.percentile(c_norms, 95))
        for u in range(args.N):
            fv = user_vecs[u].reshape(1, -1)
            user_vecs[u] = clip_rows_to_norm(fv, C_v).reshape(args.K, args.d)
            user_cnts[u] = clip_rows_to_norm(user_cnts[u].reshape(1, -1), C_c).reshape(args.K)

        # quantization bounds cover the observed magnitudes
        B_v = max(float(np.max([np.max(np.abs(v)) for v in user_vecs])), 1e-6)
        B_c = max(float(np.max([np.max(np.abs(c)) for c in user_cnts])), 1e-6)

        # ----- clean (noise-free) aggregate -----
        sum_v = np.sum(user_vecs, axis=0)          # (K,d)
        sum_c = np.sum(user_cnts, axis=0)          # (K,)
        nonempty = sum_c > 0.5
        clean_cent = np.zeros((args.K, args.d), dtype=np.float32)
        clean_cent[nonempty] = (sum_v[nonempty] / sum_c[nonempty, None]).astype(np.float32)
        clean_cent = normalize(clean_cent)

        # centroid topic labels (from clean assignment; data-independent buckets)
        cent_topic = np.zeros(args.K, dtype=int)
        for k in range(args.K):
            docs_k = np.where(bucket_of == k)[0]
            if len(docs_k):
                cent_topic[k] = np.bincount(ytr[docs_k], minlength=n_topics).argmax()

        keep = nonempty
        topic_clean_all.append(topic_acc(q_emb, q_lab, clean_cent[keep], cent_topic[keep]))

        # raw-doc 1-NN topic upper bound (no aggregation, no privacy)
        nn = np.argmax(q_emb @ Etr.T, axis=1)
        topic_rawknn_all.append(float(np.mean(ytr[nn] == q_lab)))

        # ----- DP aggregates over sigma sweep -----
        for lab, sigma in sigma_specs:
            if sigma == 0.0:
                metrics[lab]["recall"].append(1.0)
                metrics[lab]["topic_dp"].append(topic_clean_all[-1])
                continue
            per_user = [[user_vecs[u], user_cnts[u].reshape(args.K, 1)] for u in range(args.N)]
            # release sum-vector and counts with matched sigma (separate bounds/sensitivities)
            dq_v = secagg_skellam_release(
                [[pu[0]] for pu in per_user], sigma, C_v, B_v, args.N, seed)[0]
            dq_c = secagg_skellam_release(
                [[pu[1]] for pu in per_user], sigma, C_c, B_c, args.N, seed)[0].reshape(args.K)
            dq_c = np.maximum(dq_c, 1.0)
            dp_cent = np.zeros((args.K, args.d), dtype=np.float32)
            dp_cent[nonempty] = (dq_v[nonempty] / dq_c[nonempty, None]).astype(np.float32)
            dp_cent = normalize(dp_cent)
            metrics[lab]["recall"].append(
                recall_at_k(q_emb, clean_cent[keep], dp_cent[keep], args.k))
            metrics[lab]["topic_dp"].append(topic_acc(q_emb, q_lab, dp_cent[keep], cent_topic[keep]))

    # ----- report -----
    def ms(xs):
        return float(np.mean(xs)), (float(np.std(xs)) if len(xs) > 1 else 0.0)

    tc_m, tc_s = ms(topic_clean_all)
    rk_m, rk_s = ms(topic_rawknn_all)
    print("Reference (no privacy):")
    print(f"  raw-doc 1-NN topic acc  = {rk_m:.3f} +/- {rk_s:.3f}   (retrieval ceiling)")
    print(f"  clean-centroid topic acc= {tc_m:.3f} +/- {tc_s:.3f}   (K={args.K} aggregate, no noise)\n")

    print(f"{'release':<18}{'sigma':>8}{'recall@'+str(args.k):>14}{'topic-acc(DP)':>18}")
    print("-" * 58)
    for lab, sigma in sigma_specs:
        r_m, r_s = ms(metrics[lab]["recall"])
        t_m, t_s = ms(metrics[lab]["topic_dp"])
        print(f"{lab:<18}{sigma:>8.3f}{r_m:>9.3f}+/-{r_s:<4.2f}{t_m:>11.3f}+/-{t_s:<4.2f}")
    print("\n(recall@k = overlap of top-k retrieved centroids, DP vs noise-free memory)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=50)
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--M", type=int, default=1000)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--eps", type=str, default="8,3")
    p.add_argument("--fl_sigma", type=float, default=2.854)
    p.add_argument("--embedder", type=str, default="tfidf", choices=["tfidf", "st"])
    p.add_argument("--seeds", type=str, default="0,1")
    run(p.parse_args())