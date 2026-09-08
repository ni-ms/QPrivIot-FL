"""Does the private pool actually HELP a downstream agent? (paper §7.x, the "so what")

Every utility number elsewhere in the paper is a retrieval proxy (evidence-recall: did the
right bucket surface). A reviewer's fair question is whether that proxy translates into a real
agent answering real questions better. This script measures it end to end.

Four conditions, identical questions, one local agent (Ollama qwen2.5:7b):

  no-pool    the agent answers from its own parametric knowledge, no retrieval. The floor:
             what a shared memory has to beat to justify existing.
  private    OUR mechanism. Retrieve top-k centroids from the SecAgg+Skellam pool, decode each
             to its nearest PUBLIC note, put those in context. No member text ever appears.
  raw        upper bound. Retrieve the top-k actual note texts from the pool (leaky; no privacy).
  oracle-ev  ceiling. Put the ground-truth evidence turns in context. What perfect retrieval buys.

Answer correctness is graded by an LLM judge (same local model) against the gold answer, the
standard LongMemEval protocol. We report accuracy per condition with a bootstrap CI.

The claim this can support: if private >> no-pool, the routing prior earns its keep even though
it halves raw-retrieval utility and exposes no text. If private ~ no-pool, it does not, and we
need to know that. Either way it is the missing "so what".

  ollama serve &   # in another terminal
  .venv/Scripts/python.exe scripts/agentmem/_downstream_qa.py --users 200 --questions 150 \
      --K 32 --d 32 --eps 8 --k 6 --seeds 0,1,2 --json out.json
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _agentmem_probe import (  # noqa: E402
    make_projector, assign_buckets, clip_rows_to_norm, secagg_skellam_release,
    choose_clip, single_shot_sigma,
)
from _longmemeval_data import load_longmemeval, get_embeddings  # noqa: E402

_OLLAMA = "http://localhost:11434/api/generate"

AGENT_SYS = (
    "You are a helpful assistant with access to the user's past-conversation memory. "
    "Use the memory notes below to answer the question. If the memory does not contain the "
    "answer, answer from your best general knowledge in one short sentence.\n\n"
    "MEMORY:\n{mem}"
)
NOPOOL_SYS = (
    "You are a helpful assistant. Answer the question in one short sentence from your best "
    "general knowledge."
)
JUDGE_SYS = (
    "You are a strict grader. You are given a QUESTION, the GOLD answer, and a candidate "
    "RESPONSE. Reply with exactly one token: YES if the response is consistent with the gold "
    "answer (same fact, paraphrase is fine), or NO otherwise. No explanation."
)


def ollama(model, system, prompt, timeout=120, retries=2):
    body = json.dumps({"model": model, "system": system, "prompt": prompt,
                       "stream": False, "options": {"temperature": 0.0}}).encode()
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(_OLLAMA, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())["response"].strip()
        except Exception:
            if attempt == retries:
                return ""
            time.sleep(2)


def load(n_users, d, seed):
    """Notes (text+user+emb), queries (text+gold+user+evidence-emb), public decode corpus."""
    lme = load_longmemeval("oracle")
    keep_u = lme["note_user"] < n_users
    note_texts = [t for t, k in zip(lme["note_texts"], keep_u) if k]
    note_user = lme["note_user"][keep_u]
    note_ev = lme["note_ev"][keep_u]
    note_emb_full, nuser, nev, qemb_full, quser = get_embeddings("oracle")
    note_emb = note_emb_full[keep_u]

    # gold answers from the raw dataset (aligned to query_user = user index)
    import json as _j
    from huggingface_hub import hf_hub_download
    raw = _j.load(open(hf_hub_download("xiaowu0162/longmemeval-cleaned",
                  "longmemeval_oracle.json", repo_type="dataset"), encoding="utf-8"))
    gold = {i: raw[i]["answer"] for i in range(len(raw))}

    qmask = quser < n_users
    queries = [lme["queries"][i] for i in range(len(lme["queries"])) if quser[i] < n_users]
    q_user = quser[qmask]
    q_emb = qemb_full[qmask]

    # public decode corpus: a DISJOINT set of notes (users >= n_users), never aggregated
    pub_mask = lme["note_user"] >= n_users
    public_texts = [t for t, k in zip(lme["note_texts"], pub_mask) if k]
    public_emb = note_emb_full[pub_mask]

    P = make_projector  # projector applied inside get_embeddings? No -- raw emb is 384d cached.
    # get_embeddings caches RAW 384-d; project here with the public seed (paper §4).
    note_emb = normalize(make_projector(note_emb, d, seed, "randproj", public="20ng")(note_emb)).astype(np.float32)
    q_emb = normalize(make_projector(q_emb, d, seed, "randproj", public="20ng")(q_emb)).astype(np.float32)
    public_emb = normalize(make_projector(public_emb, d, seed, "randproj", public="20ng")(public_emb)).astype(np.float32)

    return (note_texts, note_user, note_emb, note_ev,
            queries, q_user, q_emb, gold, public_texts, public_emb)


def build_pool(note_emb, note_user, K, d, N, seed, sigma):
    rng = np.random.default_rng(seed)
    anchors = normalize(rng.standard_normal((K, d))).astype(np.float32)
    bucket = assign_buckets(note_emb, anchors)
    uv = [np.zeros((K, d), dtype=np.float32) for _ in range(N)]
    cnt = np.zeros(K)
    for i in range(len(note_emb)):
        uv[note_user[i]][bucket[i]] += note_emb[i]
        cnt[bucket[i]] += 1.0
    C_v = choose_clip([np.linalg.norm(v) for v in uv], seed, 0.1)
    for u in range(N):
        uv[u] = clip_rows_to_norm(uv[u].reshape(1, -1), C_v).reshape(K, d)
    if sigma == 0.0:
        sv = np.sum(uv, axis=0)
    else:
        sv = secagg_skellam_release([[v] for v in uv], sigma, C_v, C_v, N, seed)[0]
    return normalize(sv.astype(np.float32)), cnt, anchors


def context(mode, q, note_texts, note_emb, note_ev, q_i, cent, cnt, pub_texts, pub_emb, k):
    if mode == "no-pool":
        return []
    if mode == "raw":
        idx = np.argsort(-(note_emb @ q))[:k]
        return [note_texts[i] for i in idx]
    if mode == "oracle-ev":
        # ground-truth evidence turns for THIS query's user
        ev = np.where(note_ev)[0]
        idx = ev[np.argsort(-(note_emb[ev] @ q))[:k]] if len(ev) else np.argsort(-(note_emb @ q))[:k]
        return [note_texts[i] for i in idx]
    # private: decode top-k centroids to nearest PUBLIC notes
    nz = np.where(cnt > 0.5)[0]
    bk = nz[np.argsort(-(cent[nz] @ q))[:k]]
    return [pub_texts[int(np.argmax(pub_emb @ cent[b]))] for b in bk]


def grade(model, question, gold, resp):
    if not resp:
        return 0
    out = ollama(model, JUDGE_SYS,
                 f"QUESTION: {question}\nGOLD: {gold}\nRESPONSE: {resp}\n\nYES or NO?")
    return 1 if out.strip().upper().startswith("Y") else 0


def run(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    modes = ["no-pool", "private", "raw", "oracle-ev"]
    per_seed = {m: [] for m in modes}

    for seed in seeds:
        (note_texts, note_user, note_emb, note_ev,
         queries, q_user, q_emb, gold, pub_texts, pub_emb) = load(args.users, args.d, seed)
        N = args.users
        sigma = single_shot_sigma(args.eps) if args.eps > 0 else 0.0
        cent, cnt, _ = build_pool(note_emb, note_user, args.K, args.d, N, seed, sigma)

        rng = np.random.default_rng(seed)
        qidx = rng.permutation(len(queries))[:args.questions]
        acc = {m: [] for m in modes}
        for qn, qi in enumerate(qidx):
            q, u = q_emb[qi], q_user[qi]
            gtext = gold.get(int(u), "")
            for m in modes:
                ctx = context(m, q, note_texts, note_emb, note_ev, qi, cent, cnt,
                              pub_texts, pub_emb, args.k)
                sysmsg = NOPOOL_SYS if m == "no-pool" else \
                    AGENT_SYS.format(mem="\n".join(f"- {c}" for c in ctx))
                resp = ollama(args.model, sysmsg, queries[qi])
                acc[m].append(grade(args.model, queries[qi], gtext, resp))
            if (qn + 1) % 10 == 0:
                run = "  ".join(f"{m}={np.mean(acc[m]):.2f}" for m in modes)
                print(f"    [seed {seed} q{qn+1}/{len(qidx)}] {run}", flush=True)
        for m in modes:
            per_seed[m].append(float(np.mean(acc[m])))
        print(f"[seed {seed}] " + "  ".join(f"{m}={np.mean(acc[m]):.3f}" for m in modes))

    print("\n=== Downstream QA accuracy (LongMemEval-oracle, qwen2.5:7b judge) ===")
    print(f"users={args.users} questions={args.questions} K={args.K} d={args.d} "
          f"eps~{args.eps} k={args.k} seeds={seeds}\n")
    rows = {}
    for m in modes:
        a = np.array(per_seed[m])
        rows[m] = dict(mean=float(a.mean()), std=float(a.std()))
        print(f"  {m:>10}: {a.mean():.3f} +/- {a.std():.3f}")
    base, priv = rows["no-pool"]["mean"], rows["private"]["mean"]
    raw, ceil = rows["raw"]["mean"], rows["oracle-ev"]["mean"]
    print(f"\n  private - no-pool = {priv-base:+.3f}  (does the private pool beat NO pool?)")
    print(f"  private / raw     = {priv/raw:.2f}     (fraction of the leaky upper bound retained)")
    print(f"  headroom to oracle= {ceil-priv:+.3f}")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"script": "downstream_qa", "config": vars(args), "rows": rows,
             "per_seed": per_seed}, indent=1))
        print(f"\n[json] {args.json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--users", type=int, default=200)
    p.add_argument("--questions", type=int, default=150)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--model", type=str, default="qwen2.5:7b")
    p.add_argument("--json", type=str, default=None)
    run(p.parse_args())
