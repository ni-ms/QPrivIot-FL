import pandas as pd, numpy as np
from scipy import stats

df = pd.read_csv("test_artifacts/metrics_long.csv")
# pivot to wide so val_accuracy is a column
w = (df[df["metric"] == "val_accuracy"]
        .pivot_table(index=["config","num_clients","alpha","seed","epsilon","round"],
                     values="value").reset_index().rename(columns={"value":"acc"}))
last = w.sort_values("round").groupby(["config","num_clients","alpha","seed","epsilon"]).tail(1)

def ci95(s):
    s = s.dropna().to_numpy()
    if len(s) < 2: return (np.nan, np.nan)
    se = stats.sem(s)
    h  = se * stats.t.ppf(0.975, len(s) - 1)
    return (s.mean() - h, s.mean() + h)

summary = (last.groupby(["config","num_clients","alpha","epsilon"])
               .agg(mean=("acc","mean"),
                    std =("acc","std"),
                    n   =("acc","size"))
               .reset_index())

# Apply CI95
cis = last.groupby(["config","num_clients","alpha","epsilon"])["acc"].apply(ci95).apply(pd.Series)
summary["ci_low"] = cis[0].values
summary["ci_high"] = cis[1].values

summary.to_csv("test_artifacts/final_accuracy_ci.csv", index=False)
print(summary)
