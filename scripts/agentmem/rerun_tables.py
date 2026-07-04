"""
Aggregate the multi-seed rerun JSON (experiment_results/rerun_grid/*.json) into
paper-ready tables: CSV (one row per config×release) + a markdown summary that mirrors
the phase-doc tables, now at whatever seed count the grid was run with (final metric,
mean +/- std). Read-only over the JSON; writes to experiment_results/rerun_grid/.

  .venv/bin/python scripts/rerun_tables.py
"""
import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "experiment_results" / "rerun_grid"


def load():
    return {p.stem: json.load(open(p)) for p in sorted(OUT.glob("*.json"))
            if not p.stem.endswith("_TABLE")}


def _row(rows, label, key):
    # clean point is labelled "inf (clean)" (longmemeval / agentmem_leakage) or "inf"
    # (agentmem_probe); accept either.
    labels = ["inf (clean)", "inf"] if label == "inf (clean)" else [label]
    for want in labels:
        for r in rows:
            if r["label"] == want:
                return r.get(key)
    return None


def _is_utility(script):
    return script.endswith("probe") or script == "distilled_utility"


def _util_keys(script):
    """(mean_key, std_key) for a utility script's headline metric."""
    if script == "agentmem_probe":
        return "topic_mean", "topic_std"
    return "mean", "std"  # longmemeval_probe, distilled_utility


def iter_leakage(data):
    """Yield (tag, script_label, cfg, lc_frac, rows) for every leakage result, expanding
    the nested distilled_leakage JSON into one entry per source (raw / distilled)."""
    for tag, d in data.items():
        if d["script"] == "distilled_leakage":
            for src, blk in d["sources"].items():
                yield (f"{tag}:{src}", f"distilled_leakage[{src}]",
                       d["config"], blk["lc_frac"], blk["rows"])
        elif "leak" in d["script"]:
            yield (tag, d["script"], d["config"], d.get("lc_frac", float("nan")), d["rows"])


def utility_csv(data, path):
    """probe / distilled_utility jsons -> long CSV (tag, config, release-point, mean, std)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "script", "K", "d", "N", "metric", "seeds", "eps_label", "mean", "std"])
        for tag, d in data.items():
            if not _is_utility(d["script"]):
                continue
            cfg = d["config"]
            mk, sk = _util_keys(d["script"])
            for r in d["rows"]:
                w.writerow([tag, d["script"], cfg["K"], cfg["d"], cfg.get("N", ""),
                            d["metric"], len(d["seeds"]), r["label"],
                            f"{r[mk]:.4f}", f"{r[sk]:.4f}"])


def leakage_csv(data, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "script", "K", "d", "N", "lc_frac", "seeds",
                    "eps_label", "auc_mean", "auc_std", "tail_auc_mean", "tail_auc_std",
                    "gap_mean"])
        for tag, script, cfg, lc_frac, rows in iter_leakage(data):
            for r in rows:
                w.writerow([tag, script, cfg["K"], cfg["d"], cfg.get("N", ""),
                            f"{lc_frac:.3f}", "", r["label"],
                            f"{r['auc_mean']:.4f}", f"{r['auc_std']:.4f}",
                            f"{r['lc_auc_mean']:.4f}", f"{r['lc_auc_std']:.4f}",
                            f"{r['gap_mean']:.4f}"])


def md_summary(data, path):
    lines = ["# Multi-seed rerun summary (final metric, mean +/- std)\n"]
    any_seeds = next(iter(data.values()))["seeds"] if data else []
    lines.append(f"Seeds: {any_seeds}  ({len(any_seeds)} seeds). "
                 f"Configs: {len(data)}.\n")

    # ---- Utility: LongMemEval oracle probe ----
    lines.append("\n## LongMemEval oracle utility — evidence-recall@5\n")
    lines.append("| tag | K | d | chance | clean | eps=16 | eps=8 | eps=3 | retention@e8 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for tag, d in sorted(data.items()):
        if d["script"] != "longmemeval_probe":
            continue
        cfg = d["config"]
        clean = _row(d["rows"], "inf (clean)", "mean")
        e16 = _row(d["rows"], "eps=16", "mean")
        e8 = _row(d["rows"], "eps=8", "mean")
        e3 = _row(d["rows"], "eps=3", "mean")
        ret = f"{100*e8/clean:.0f}%" if clean else "-"
        lines.append(f"| {tag} | {cfg['K']} | {cfg['d']} | {d['chance']:.3f} | "
                     f"{clean:.3f} | {e16:.3f} | {e8:.3f} | {e3:.3f} | {ret} |")

    # ---- Utility: synthetic/real agentmem probe (topic-acc) ----
    lines.append("\n## Agent-memory probe utility — topic-acc(DP)\n")
    lines.append("| tag | embedder | N | K | d | clean | eps=16 | eps=8 | eps=3 | retention@e8 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for tag, d in sorted(data.items()):
        if d["script"] != "agentmem_probe":
            continue
        cfg = d["config"]
        clean = _row(d["rows"], "inf (clean)", "topic_mean")
        e16 = _row(d["rows"], "eps=16", "topic_mean")
        e8 = _row(d["rows"], "eps=8", "topic_mean")
        e3 = _row(d["rows"], "eps=3", "topic_mean")
        ret = f"{100*e8/clean:.0f}%" if clean else "-"
        lines.append(f"| {tag} | {d['embedder']} | {cfg['N']} | {cfg['K']} | {cfg['d']} | "
                     f"{clean:.3f} | {'-' if e16 is None else f'{e16:.3f}'} | "
                     f"{e8:.3f} | {e3:.3f} | {ret} |")

    # ---- Utility: distilled answer-retrieval ----
    lines.append("\n## Distilled-notes utility — answer-recall@5 (vector-only release)\n")
    lines.append("| tag | N | K | d | chance | clean | eps=16 | eps=8 | eps=3 | retention@e8 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for tag, d in sorted(data.items()):
        if d["script"] != "distilled_utility":
            continue
        cfg = d["config"]
        clean = _row(d["rows"], "inf (clean)", "mean")
        e16 = _row(d["rows"], "eps=16", "mean")
        e8 = _row(d["rows"], "eps=8", "mean")
        e3 = _row(d["rows"], "eps=3", "mean")
        ret = f"{100*e8/clean:.0f}%" if clean else "-"
        lines.append(f"| {tag} | {d['stats']['N']} | {cfg['K']} | {cfg['d']} | {d['chance']:.3f} | "
                     f"{clean:.3f} | {'-' if e16 is None else f'{e16:.3f}'} | {e8:.3f} | {e3:.3f} | {ret} |")

    # ---- Leakage: all leakage scripts (distilled_leakage expanded per source) ----
    lines.append("\n## Leakage-drop — MIA tail-AUC (0.5 = no leakage)\n")
    lines.append("| tag | script | K | d | tail% | clean tail-AUC | eps=8 tail-AUC | eps=3 tail-AUC | clean all-AUC |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for tag, script, cfg, lc_frac, rows in sorted(iter_leakage(data), key=lambda t: t[0]):
        ct = _row(rows, "inf (clean)", "lc_auc_mean")
        e8 = _row(rows, "eps=8", "lc_auc_mean")
        e3 = _row(rows, "eps=3", "lc_auc_mean")
        ca = _row(rows, "inf (clean)", "auc_mean")
        lines.append(f"| {tag} | {script} | {cfg['K']} | {cfg['d']} | "
                     f"{100*lc_frac:.0f}% | {ct:.3f} | {e8:.3f} | {e3:.3f} | {ca:.3f} |")

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    data = load()
    if not data:
        print("no JSON found in", OUT)
        return
    utility_csv(data, OUT / "utility_TABLE.csv")
    leakage_csv(data, OUT / "leakage_TABLE.csv")
    md_summary(data, OUT / "SUMMARY_TABLE.md")
    print(f"aggregated {len(data)} configs ->")
    print(f"  {OUT/'utility_TABLE.csv'}")
    print(f"  {OUT/'leakage_TABLE.csv'}")
    print(f"  {OUT/'SUMMARY_TABLE.md'}")


if __name__ == "__main__":
    main()
