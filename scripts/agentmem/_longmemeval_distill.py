"""
LLM memory-note distillation for LongMemEval — A-MEM / Mem0 style, via a LOCAL model
(Ollama, free). Turns each raw dialogue session into a handful of atomic "memory notes"
(facts / events / preferences), which is what real agent-memory systems store instead of
raw turns. Output feeds the same SecAgg+Skellam DP centroid-memory pipeline.

Local + free: no API key, no data leaves the machine (apt for a privacy paper).
Requires `ollama serve` running and a pulled model (default qwen2.5:7b).

Usage:
  .venv/bin/python scripts/_longmemeval_distill.py --limit 50 --model qwen2.5:7b
  .venv/bin/python scripts/_longmemeval_distill.py            # full oracle (background)
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

_REPO = "xiaowu0162/longmemeval-cleaned"
_FILES = {"oracle": "longmemeval_oracle.json", "s": "longmemeval_s_cleaned.json"}
_OUT = Path(__file__).resolve().parents[2] / "experiment_results"
_OLLAMA = "http://localhost:11434/api/generate"

PROMPT = """You are a memory-extraction module for a personal AI assistant. From the
conversation below, extract the durable MEMORY NOTES worth storing about the user: facts,
events, preferences, plans, and relationships. Write each as ONE short standalone sentence
(no pronouns like "they" without a referent). Output ONLY the notes, one per line, no
numbering, no preamble. If nothing is worth remembering, output nothing.

CONVERSATION:
{conv}

MEMORY NOTES:"""


def ollama(model, prompt, timeout=120):
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.2, "num_predict": 400}}).encode()
    req = urllib.request.Request(_OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["response"]


def session_text(sess, max_turns=40):
    lines = []
    for t in sess[:max_turns]:
        role = t.get("role", "user")
        content = (t.get("content") or "").strip().replace("\n", " ")
        if content:
            lines.append(f"{role}: {content[:600]}")
    return "\n".join(lines)


def run(args):
    path = hf_hub_download(_REPO, _FILES[args.variant], repo_type="dataset")
    data = json.load(open(path, encoding="utf-8"))
    if args.limit:
        data = data[:args.limit]

    cache = _OUT / f"_lme_distilled_{args.variant}{'_pilot' if args.limit else ''}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)

    def save(obj):  # checkpoint so a long run survives an interruption / crash
        json.dump(obj, open(cache, "w", encoding="utf-8"))

    out = []
    n_notes = 0
    for ui, ex in enumerate(data):
        user_notes = []
        for si, sess in enumerate(ex["haystack_sessions"]):
            conv = session_text(sess)
            if not conv:
                continue
            try:
                resp = ollama(args.model, PROMPT.format(conv=conv))
            except Exception as e:
                print(f"  [user {ui} sess {si}] LLM error: {repr(e)[:80]}", file=sys.stderr)
                continue
            notes = [ln.strip("-•* \t") for ln in resp.splitlines() if len(ln.strip()) > 8]
            for note in notes:
                user_notes.append(note)
                n_notes += 1
        out.append({"user": ui, "question": ex["question"], "notes": user_notes})
        if (ui + 1) % 10 == 0:
            save(out)  # incremental checkpoint
            print(f"distilled {ui+1}/{len(data)} users, {n_notes} notes so far", flush=True)

    save(out)
    print(f"\nDONE: {len(out)} users, {n_notes} distilled notes -> {cache}")
    # sample
    for ex in out[:2]:
        print(f"\n--- user {ex['user']} Q: {ex['question'][:70]}")
        for note in ex["notes"][:6]:
            print(f"   - {note[:100]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", type=str, default="oracle", choices=["oracle", "s"])
    p.add_argument("--model", type=str, default="qwen2.5:7b")
    p.add_argument("--limit", type=int, default=0, help="0 = all users")
    run(p.parse_args())