import pandas as pd
from scipy import stats

df = pd.read_csv("test_artifacts/metrics_long.csv")
w  = df[df["metric"] == "val_accuracy"].pivot_table(
        index=["config","num_clients","alpha","seed","epsilon","round"],
        values="value").reset_index().rename(columns={"value":"acc"})
final = w.sort_values("round").groupby(
            ["config","num_clients","alpha","seed","epsilon"]).tail(1)

def pair(a, b, **fixed):
    sub = final
    for k, v in fixed.items(): sub = sub[sub[k] == v]
    aa = sub[sub["config"] == a].sort_values("seed")["acc"].to_numpy()
    bb = sub[sub["config"] == b].sort_values("seed")["acc"].to_numpy()
    if len(aa) < 2 or len(aa) != len(bb): return None
    t, p_t = stats.ttest_rel(aa, bb)
    # Wilcoxon requires differences to be non-zero usually, handle zero differences
    diff = aa - bb
    if np.all(diff == 0):
        w_stat, p_w = 0.0, 1.0
    else:
        w_stat, p_w = stats.wilcoxon(aa, bb, zero_method="wilcox", alternative="two-sided")
    
    return dict(t=float(t), p_t=float(p_t),
                wilcoxon=float(w_stat), p_w=float(p_w),
                mean_diff=float(aa.mean() - bb.mean()), n=len(aa))

import numpy as np
# Example call
res = pair("adapriv", "fixed-dp", num_clients=10, alpha=0.3, epsilon=3.0)
if res:
    print(f"Comparison adapriv vs fixed-dp: {res}")
else:
    print("Not enough data for comparison (needs matched seeds).")
