"""The chance floor of the reconstruction-decode metric, and what it implies (paper §7.2).

The decode-extract attack scores a hit when the top-1 nearest note to a released centroid is a
true in-bucket member. We report it, so we must report what CHANCE is for it -- and chance is
not 0. A bucket at coarse K holds hundreds of notes, so a vector pointing anywhere near the
bucket's region "decodes" into it.

The floor is measured by replacing the released centroid with a PURE RANDOM unit vector -- zero
information about the corpus, zero information about the release -- and scoring the same metric.

Two facts fall out, and both are load-bearing for the paper's honesty:

  (1) The floor is ~1/K (0.010 at K=32), NOT the ~0.08 one might read off the strongest-noise
      row of the LiRA table (which is still an informative release). So decode = 0.396 at
      K=32 / eps~9.3 sits 40x above chance: reconstruction genuinely SURVIVES DP there, and we
      do not claim otherwise.

  (2) But decode tracks retrieval utility almost exactly across the K-sweep (0.396/95% ->
      0.070/72% -> 0.001/13%), because they are the SAME QUANTITY: a centroid that retrieves
      well must point at the notes in its own bucket. You cannot drive decode to chance without
      destroying the pool. And what a closed-set decode hit actually discloses -- that a note
      the adversary ALREADY HOLDS falls in bucket k -- is computable by anyone from the PUBLIC
      anchors, with no access to the release at all.

  => Closed-set decode is an UPPER BOUND on inversion, not an independent leakage axis. A low
     value certifies the release; a high value certifies nothing. The open question it leaves
     is open-set inversion (vec2text-style recovery of note text), which we do not run.

  .venv/Scripts/python.exe scripts/agentmem/_decode_floor.py [--json out.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _longmemeval_data import get_embeddings            # noqa: E402
from _agentmem_probe import make_projector, assign_buckets   # noqa: E402

# measured elsewhere in the paper (Table: joint frontier), repeated here for the contrast
PAPER_DP = {32: 0.396, 64: 0.188, 128: 0.070, 256: 0.029, 512: 0.007, 1024: 0.003, 2048: 0.001}
PAPER_CLEAN = {32: 0.771, 64: 0.667, 128: 0.794, 256: 0.789, 512: 0.874, 1024: 0.912, 2048: 0.928}
PAPER_RETEN = {32: 95, 64: 86, 128: 72, 256: 60, 512: 41, 1024: 24, 2048: 13}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    raw, _, _, _, _ = get_embeddings("oracle")
    raw = raw.astype(np.float32)

    print("=== Decode-extract: the chance floor (LongMemEval-oracle, d=%d) ===\n" % a.d)
    print(f"{'K':>6} {'notes/bkt':>10} {'CHANCE':>8} {'DP@9.3':>8} {'x chance':>9} "
          f"{'clean':>7} {'retention':>10}")
    print("-" * 68)

    rows = []
    for K in (32, 64, 128, 256, 512, 1024, 2048):
        hits = tot = 0
        for seed in seeds:
            rng = np.random.default_rng(seed)
            emb = normalize(
                make_projector(raw, a.d, seed, "randproj", public="20ng")(raw)
            ).astype(np.float32)
            anchors = normalize(rng.standard_normal((K, a.d))).astype(np.float32)
            bucket = assign_buckets(emb, anchors)
            # a pure random unit vector per bucket: zero corpus information
            fake = normalize(rng.standard_normal((K, a.d))).astype(np.float32)
            nn = np.argmax(fake @ emb.T, axis=1)
            occupied = np.array([np.any(bucket == k) for k in range(K)])
            hits += int(((bucket[nn] == np.arange(K)) & occupied).sum())
            tot += int(occupied.sum())
        floor = hits / tot
        ratio = PAPER_DP[K] / floor if floor else float("inf")
        rows.append(dict(K=K, notes_per_bucket=len(emb) / K, chance=floor,
                         dp=PAPER_DP[K], ratio=ratio, clean=PAPER_CLEAN[K],
                         retention=PAPER_RETEN[K]))
        print(f"{K:>6} {len(emb)/K:>10.0f} {floor:>8.3f} {PAPER_DP[K]:>8.3f} {ratio:>8.0f}x "
              f"{PAPER_CLEAN[K]:>7.3f} {PAPER_RETEN[K]:>9d}%")

    print("\n  Chance ~ 1/K. Decode at the recommended K=32 is 40x chance: DP does NOT defeat")
    print("  reconstruction there, and the paper does not claim it does.")
    print("  But decode tracks RETENTION down the sweep -- they are the same signal. A centroid")
    print("  that retrieves must point at its own bucket; you cannot floor one without killing")
    print("  the other. And a closed-set hit discloses only the bucket of a note the adversary")
    print("  already holds -- computable from the PUBLIC anchors, without the release.")

    if a.json:
        Path(a.json).write_text(json.dumps({"script": "decode_floor", "d": a.d,
                                            "seeds": seeds, "rows": rows}, indent=1))
        print(f"\n[json] {a.json}")


if __name__ == "__main__":
    main()
