"""
LongMemEval loader for the federated agent-memory experiments (Phase 2, real data).

Maps the benchmark onto our federated setting:
  - USER        = one question's haystack (one user's chat history); 500 users.
  - MEMORY NOTE = one dialogue turn's content (a unit stored in agent memory).
  - QUERY       = the question.
  - GROUND TRUTH= turns flagged has_answer (the evidence a correct retriever must surface).

Provides real embeddings (cached all-MiniLM-L6-v2) so the DP centroid-memory pipeline can be
evaluated with a REAL retrieval objective (evidence recall) instead of a topic proxy.

Source: xiaowu0162/longmemeval-cleaned  (oracle = evidence-only haystacks, compact;
        s = full haystacks with distractors, ~25x larger).
"""
import json
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

_REPO = "xiaowu0162/longmemeval-cleaned"
_FILES = {"oracle": "longmemeval_oracle.json", "s": "longmemeval_s_cleaned.json"}
_CACHE_DIR = Path(__file__).resolve().parents[2] / "experiment_results"


def load_longmemeval(variant="oracle", min_chars=8):
    """Return dict: note_texts, note_user(int[]), note_ev(bool[]), queries, query_user(int[])."""
    path = hf_hub_download(_REPO, _FILES[variant], repo_type="dataset")
    data = json.load(open(path))
    note_texts, note_user, note_ev = [], [], []
    queries, query_user = [], []
    for ui, ex in enumerate(data):
        queries.append(ex["question"])
        query_user.append(ui)
        for sess in ex["haystack_sessions"]:
            for t in sess:
                c = (t.get("content") or "").strip()
                if len(c) < min_chars:
                    continue
                note_texts.append(c)
                note_user.append(ui)
                note_ev.append(bool(t.get("has_answer")))
    return {
        "note_texts": note_texts,
        "note_user": np.array(note_user),
        "note_ev": np.array(note_ev, dtype=bool),
        "queries": queries,
        "query_user": np.array(query_user),
    }


def get_embeddings(variant="oracle", model_name="all-MiniLM-L6-v2"):
    """Encode notes + queries once with a REAL sentence encoder; cache to disk.
    Returns (note_emb[N,384], note_user, note_ev, query_emb[Q,384], query_user)."""
    cache = _CACHE_DIR / f"_st_lme_{variant}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["note_emb"], z["note_user"], z["note_ev"], z["query_emb"], z["query_user"]
    d = load_longmemeval(variant)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name)
    note_emb = m.encode(d["note_texts"], batch_size=256, show_progress_bar=True,
                        normalize_embeddings=False).astype(np.float32)
    query_emb = m.encode(d["queries"], batch_size=256, show_progress_bar=True,
                         normalize_embeddings=False).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, note_emb=note_emb, note_user=d["note_user"], note_ev=d["note_ev"],
             query_emb=query_emb, query_user=d["query_user"])
    return note_emb, d["note_user"], d["note_ev"], query_emb, d["query_user"]


if __name__ == "__main__":
    ne, nu, nev, qe, qu = get_embeddings("oracle")
    print(f"notes={ne.shape} users={nu.max()+1} evidence={nev.sum()} queries={qe.shape}")