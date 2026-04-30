import pandas as pd
df = pd.read_csv("test_artifacts/metrics_long.csv")
w  = df[df["metric"] == "val_accuracy"].pivot_table(
        index=["config","num_clients","alpha","seed","epsilon","round"],
        values="value").reset_index().rename(columns={"value":"acc"})

TARGETS = [0.40, 0.50, 0.60, 0.70]
out = []
for tgt in TARGETS:
    for keys, g in w.groupby(["config","num_clients","alpha","seed","epsilon"]):
        g = g.sort_values("round")
        hit = g[g["acc"] >= tgt]
        out.append({"config": keys[0], "num_clients": keys[1], "alpha": keys[2],
                    "seed": keys[3], "epsilon": keys[4], "target": tgt,
                    "rounds_to": int(hit["round"].iloc[0]) if len(hit)
                                 else int(g["round"].max()+1),
                    "reached": bool(len(hit))})
pd.DataFrame(out).to_csv("test_artifacts/convergence_rate.csv", index=False)
print("Wrote test_artifacts/convergence_rate.csv")
