import pandas as pd, numpy as np
from scipy.optimize import curve_fit

df = pd.read_csv("test_artifacts/metrics_long.csv")
w  = df[df["metric"] == "val_accuracy"].pivot_table(
        index=["config","num_clients","alpha","seed","epsilon","round"],
        values="value").reset_index().rename(columns={"value":"acc"})

def model(t, a_inf, a_0, tau):  return a_inf - (a_inf - a_0) * np.exp(-t / tau)

rows = []
for keys, g in w.groupby(["config","num_clients","alpha","seed","epsilon"]):
    g = g.sort_values("round")
    try:
        (a_inf, a_0, tau), _ = curve_fit(model, g["round"], g["acc"],
                                         p0=(0.7, 0.1, 10.0), maxfev=5000)
        rows.append({"config": keys[0], "num_clients": keys[1], "alpha": keys[2],
                     "seed": keys[3], "epsilon": keys[4],
                     "a_inf": a_inf, "a_0": a_0, "tau": tau})
    except Exception as e:
        print("fit failed:", keys, e)
pd.DataFrame(rows).to_csv("test_artifacts/convergence_tau.csv", index=False)
print("Wrote test_artifacts/convergence_tau.csv")
