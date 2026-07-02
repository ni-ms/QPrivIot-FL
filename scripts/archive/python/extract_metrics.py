"""
Reads:
  histories/*.pkl                                   (Flower History objects)
  experiment_results/results_*_n*_alpha*_seed*_eps*.json   (project JSON)
Writes:
  test_artifacts/metrics_long.csv

Schema (long form):
  dataset, config, num_clients, alpha, seed, epsilon, round, metric, value, source
  source ∈ {centralized, distributed_eval, distributed_fit, project_json}
"""
from __future__ import annotations
import argparse, glob, json, pickle, re, pathlib, sys
import pandas as pd

NAME_RE = re.compile(
    r"results_(?P<dataset>\w+)_(?P<config>[\w-]+)(?:_n(?P<n>\d+))?(?:_alpha(?P<alpha>[\d.]+))?"
    r"_seed(?P<seed>\d+)_eps(?P<eps>[\d.]+)\.json"
)

def _parse_filename(path: str):
    m = NAME_RE.search(pathlib.Path(path).name)
    if not m:
        return {"dataset": "cifar10", "config": "unknown", "num_clients": 10, "alpha": 0.3, "seed": 42, "epsilon": 3.0}
    
    return {
        "dataset": m.group("dataset"),
        "config": m.group("config"),
        "num_clients": int(m.group("n")) if m.group("n") else 10,
        "alpha": float(m.group("alpha")) if m.group("alpha") else 0.3,
        "seed": int(m.group("seed")),
        "epsilon": float(m.group("eps"))
    }

def _from_history_pkl(path: str) -> pd.DataFrame:
    h = pickle.load(open(path, "rb"))
    rows: list[dict] = []
    for r, loss in h.losses_centralized:
        rows.append(dict(round=r, metric="loss", value=loss, source="centralized"))
    for r, loss in h.losses_distributed:
        rows.append(dict(round=r, metric="loss", value=loss, source="distributed_eval"))
    for k, series in h.metrics_centralized.items():
        for r, v in series:
            rows.append(dict(round=r, metric=k, value=float(v), source="centralized"))
    for k, series in h.metrics_distributed.items():
        for r, v in series:
            rows.append(dict(round=r, metric=k, value=float(v), source="distributed_eval"))
    for k, series in h.metrics_distributed_fit.items():
        for r, v in series:
            rows.append(dict(round=r, metric=k, value=float(v), source="distributed_fit"))
    df = pd.DataFrame(rows)
    df["pkl_path"] = path
    return df

def _from_project_json(path: str) -> pd.DataFrame:
    payload = json.load(open(path))
    rows = []
    for entry in payload["rounds"]:
        r = entry["round"]
        for k, v in entry.items():
            if k == "round" or not isinstance(v, (int, float, bool)):
                continue
            rows.append(dict(round=r, metric=k, value=float(v),
                             source="project_json"))
    df = pd.DataFrame(rows)
    meta = _parse_filename(path)
    if meta:
        for k, val in meta.items():
            df[k] = val
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-glob", default="experiment_results/results_*.json")
    p.add_argument("--history-glob", default="histories/*.pkl")
    p.add_argument("--out", default="test_artifacts/metrics_long.csv")
    args = p.parse_args()

    frames: list[pd.DataFrame] = []
    for f in glob.glob(args.results_glob):
        df = _from_project_json(f)
        if not df.empty:
            frames.append(df)
    for f in glob.glob(args.history_glob):
        frames.append(_from_history_pkl(f))

    if not frames:
        sys.exit("No results found.")

    out = pd.concat(frames, ignore_index=True, sort=False)

    # Acceptance gate: drop NaN rows
    bad = out["value"].isna().sum()
    if bad:
        print(f"⚠️  Dropping {bad} NaN rows", file=sys.stderr)
        out = out.dropna(subset=["value"])

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out):,} rows to {args.out}")

if __name__ == "__main__":
    main()
