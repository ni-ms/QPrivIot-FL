"""
Distilled-notes UTILITY endpoint (fills the gap left by distillation dropping per-turn
has_answer labels). Metric = ANSWER RETRIEVAL: for each question, identify the user's
answer-bearing note (the distilled note most similar to the gold answer), then measure
whether the DP centroid memory routes the question to that note's bucket.

  answer-recall@k = fraction of questions whose answer-bearing note's bucket is in the
                    question's top-k retrieved centroids.

Uses the vector-only DP release (1 RDP composition, the recommended design). Reports clean
vs DP over an eps sweep. Local + free; reuses the distilled embedding cache.

Usage:
  .venv/bin/python scripts/_longmemeval_distilled_utility.py --distilled experiment_results/_lme_distilled_oracle_pilot.json --K 64 --topk 5 --seeds 0,1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _agentmem_probe import assign_buckets, clip_rows_to_norm, secagg_skellam_release, single_shot_sigma  # noqa: E402

_CACHE = Path(__file__).resolve().parents[1] / "experiment_results"


def load_distilled_emb(distilled_path):
    cache = _CACHE / (Path(distilled_path).stem + "_emb.npz")
    z = np.load(cache)
    return z["emb"], z["user"]


def encode_qa(n_users, variant="oracle"):
    path = hf_hub_download("xiaowu0162/longmemeval-cleaned",
                           "longmemeval_oracle.json" if variant == "oracle" else "longmemeval_s_cleaned.json",
                           repo_type="dataset")
    data = json.load(open(path))[:n_users]
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-MiniLM-L6-v2")
    q = m.encode([d["question"] for d in data], normalize_embeddings=False).astype(np.float32)
    a = m.encode([str(d["answer"]) for d in data], normalize_embeddings=False).astype(np.float32)
    return q, a


def vec_release(user_vecs, K, d, N, C_v, B_v, sigma, seed):
    if sigma == 0.0:
        return np.sum(user_vecs, axis=0)
    return secagg_skellam_release([[v] for v in user_vecs], sigma, C_v, B_v, N, seed)[0]


def run(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    eps_points = [float(e) for e in args.eps.split(",")]
    sigma_specs = [("inf (clean)", 0.0)] + [(f"eps={e:g}", single_shot_sigma(e)) for e in eps_points]
    sigma_specs.append(("FL-sigma", args.fl_sigma))

    note_raw, note_user = load_distilled_emb(args.distilled)
    N = int(note_user.max() + 1)
    q_raw, a_raw = encode_qa(N)

    print(f"=== Distilled-notes ANSWER-RETRIEVAL utility (K={args.K}, d={args.d}, {N} users) ===")
    print(f"notes={len(note_raw)} | top-{args.topk} | chance~={args.topk/args.K:.3f}\n")

    metrics = {lab: [] for lab, _ in sigma_specs}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        pca = PCA(n_components=args.d, random_state=seed)
        note_emb = normalize(pca.fit_transform(note_raw)).astype(np.float32)
        q_emb = normalize(pca.transform(q_raw)).astype(np.float32)
        a_emb = normalize(pca.transform(a_raw)).astype(np.float32)
        anchors = normalize(rng.standard_normal((args.K, args.d))).astype(np.float32)
        bucket = assign_buckets(note_emb, anchors)

        # per-user answer-bearing note = note most similar to the gold answer
        target_bucket = np.full(N, -1)
        for u in range(N):
            idx = np.where(note_user == u)[0]
            if len(idx) == 0:
                continue
            best = idx[np.argmax(note_emb[idx] @ a_emb[u])]
            target_bucket[u] = bucket[best]

        uv = [np.zeros((args.K, args.d)) for _ in range(N)]
        uc = [np.zeros(args.K) for _ in range(N)]
        for i in range(len(note_emb)):
            uv[note_user[i]][bucket[i]] += note_emb[i]; uc[note_user[i]][bucket[i]] += 1.0
        C_v = float(np.percentile([np.linalg.norm(v) for v in uv], 95))
        for u in range(N):
            uv[u] = clip_rows_to_norm(uv[u].reshape(1, -1), C_v).reshape(args.K, args.d)
        B_v = max(float(np.max([np.max(np.abs(v)) for v in uv])), 1e-6)
        nonempty = np.sum(uc, axis=0) > 0.5

        valid = target_bucket >= 0
        for lab, sigma in sigma_specs:
            sv = vec_release(uv, args.K, args.d, N, C_v, B_v, sigma, seed)
            cent = np.zeros((args.K, args.d), dtype=np.float32)
            cent[nonempty] = sv[nonempty].astype(np.float32)
            cent = normalize(cent)
            topk = np.argsort(-(q_emb @ cent.T), axis=1)[:, :args.topk]
            hits = [target_bucket[u] in topk[u] for u in range(N) if valid[u]]
            metrics[lab].append(float(np.mean(hits)))

    print(f"{'release':<14}{'sigma':>8}{'answer-recall@'+str(args.topk):>20}")
    print("-" * 42)
    for lab, sigma in sigma_specs:
        xs = metrics[lab]
        print(f"{lab:<14}{sigma:>8.3f}{np.mean(xs):>14.3f} +/-{(np.std(xs) if len(xs)>1 else 0):<5.2f}")
    print(f"\nchance~={args.topk/args.K:.3f}. Does clean >> chance and DP hold up?")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--distilled", type=str, required=True)
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--eps", type=str, default="16,8,3")
    p.add_argument("--fl_sigma", type=float, default=2.854)
    p.add_argument("--seeds", type=str, default="0,1")
    run(p.parse_args())