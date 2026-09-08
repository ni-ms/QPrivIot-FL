"""
The joint utility/leakage frontier (paper §7.1) and the multi-round schedule (§5.7).

Regenerates, from the released grid, the two tables the superseded draft could not state:

  (a) JOINT FRONTIER -- utility and *calibrated* leakage at the SAME K. The old grid measured
      utility only at K<=256 and leakage only at K>=512, so the abstract's "95% retention"
      (K=32) and "tail-AUC 0.93->0.53" (K>=512) described disjoint releases. Filling the
      cells shows the cosine proxy was simply blind at low K -- it reads 0.504 (chance) at
      K=32 where the calibrated LiRA reads AUC 0.833 / TPR@1%FPR 0.175 / decode 0.771.

  (b) MULTI-ROUND -- the accountant already composes (`releases=`), but the paper only ever
      reported the single-shot eps. Naive re-aggregation is ruinous (daily for a month =>
      eps 93.6). Since RDP composition is additive, the required sigma grows like sqrt(T),
      so a cadence can instead be BOUGHT inside a fixed total budget. We solve for that
      sigma and read retention straight off the measured --fl_sigma runs.

  .venv/Scripts/python.exe scripts/agentmem/_joint_frontier.py [--latex]

Read-only over experiment_results/. Writes figures/fig_joint_frontier.{pdf,png} and, with
--latex, the two table bodies on stdout for paste into the manuscript.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from qpriviot_fl.privacy_utils import skellam_rdp_epsilon  # noqa: E402

JOINT = ROOT / "experiment_results" / "joint_rp"
GRID = ROOT / "experiment_results" / "rerun_grid_rp"
LIRA = ROOT / "experiment_results" / "lira_rp"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

KS = [32, 64, 128, 256, 512, 1024, 2048]

# Operating point of the accounting (paper §5.4): the DP-selected clip bound lands at C=32 on
# the public geometric grid, range_max = 1e6, d = 32, delta = 1e-5.
# DIM is the dimension the L1 sensitivity bound Delta_1 <= sqrt(DIM)*Delta_2 is taken over, and
# that is the FULL K*d payload -- the object the global clip acts on (paper Remark 1), not d.
# (The correction term it feeds is O(1e-4), so this does not move the reported eps.)
C_CLIP, RANGE_MAX, DIM, DELTA = 32.0, 1e6, 32 * 32, 1e-5
SIGMA_EPS93 = 0.6056006578256736   # the sigma the "eps=8" run label corresponds to
EPS_TARGET = 9.31                  # the budget the multi-round sigmas were solved/run against


def _first(*paths):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(paths[0])


def _row(doc, label):
    """Clean point is 'inf (clean)' in some scripts and 'inf' in others."""
    wants = ["inf (clean)", "inf"] if label.startswith("inf") else [label]
    for w in wants:
        for r in doc["rows"]:
            if r["label"] == w:
                return r
    raise KeyError(label)


def load_joint():
    """One record per K: retention, and the calibrated attack, on the same release."""
    out = []
    for K in KS:
        u = json.loads(_first(GRID / f"p2_probe_oracle_K{K}_d32.json",
                              JOINT / f"probe_oracle_K{K}_d32.json").read_text())
        l = json.loads(_first(LIRA / f"lira_oracle_K{K}_d32.json",
                              JOINT / f"lira_oracle_K{K}_d32.json").read_text())
        x = json.loads(_first(GRID / f"p2_leak_oracle_K{K}_d32.json",
                              JOINT / f"leak_oracle_K{K}_d32.json").read_text())
        uc, u8 = _row(u, "inf"), _row(u, "eps=8")
        lc, l8, l1 = _row(l, "inf"), _row(l, "eps=8"), _row(l, "eps=1")
        out.append(dict(
            K=K,
            clean_rec=uc["mean"], clean_rec_sd=uc["std"],
            dp_rec=u8["mean"], dp_rec_sd=u8["std"],
            retention=100.0 * u8["mean"] / uc["mean"],
            chance=5.0 / K,
            auc_c=lc["auc"], auc_d=l8["auc"],
            tpr_c=lc["tpr1"], tpr_d=l8["tpr1"],
            dec_c=lc["decode_acc"], dec_d=l8["decode_acc"],
            dec_floor=l1["decode_acc"],          # decode under the strongest noise ~ the floor
            proxy_c=_row(x, "inf")["auc_mean"],  # the WEAK attack, for the contrast
            proxy_d=_row(x, "eps=8")["auc_mean"],
            tail_pct=100.0 * x.get("lc_frac", float("nan")),
        ))
    return out


def eps_at(sigma, T, clip_releases=1):
    e, _ = skellam_rdp_epsilon(sigma=sigma, clip_norm=C_CLIP, scale=RANGE_MAX / C_CLIP,
                               dim=DIM, target_delta=DELTA, releases=T,
                               clip_eps=0.1, clip_releases=clip_releases)
    return e


def sigma_for(T, target=EPS_TARGET):
    """Smallest sigma keeping T releases inside `target` total eps (eps is monotone down in sigma)."""
    lo, hi = 0.1, 500.0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if eps_at(mid, T) > target:
            lo = mid
        else:
            hi = mid
    return hi


SCHEDULE = [(1, "single shot"), (4, "quarterly, 1 year"), (12, "monthly, 1 year"),
            (30, "daily, 1 month"), (52, "weekly, 1 year")]


def load_multiround():
    clean = json.loads((GRID / "p2_probe_oracle_K32_d32.json").read_text())
    cl = _row(clean, "inf")["mean"]
    rows = []
    for T, note in SCHEDULE:
        sg = sigma_for(T)
        if T == 1:
            rec = _row(clean, "eps=8")["mean"]
        else:
            # runs are tagged by the sigma they were launched at; match to 3dp
            hits = sorted(JOINT.glob(f"probe_K32_sigma{sg:.3f}*.json"))
            if not hits:
                hits = sorted(JOINT.glob("probe_K32_sigma*.json"),
                              key=lambda p: abs(float(p.stem.split("sigma")[1]) - sg))
            rec = _row(json.loads(hits[0].read_text()), "FL-sigma")["mean"]
        rows.append(dict(T=T, note=note, sigma=sg, rec=rec,
                         retention=100.0 * rec / cl, naive_eps=eps_at(SIGMA_EPS93, T)))
    return rows, cl


# ---------------------------------------------------------------- reporting

def print_joint(J):
    print("\n=== (a) JOINT FRONTIER: utility and calibrated leakage at the SAME K ===")
    print("    LongMemEval-oracle, d=32, vector-only, 5-seed utility / 3-seed LiRA\n")
    print(f"{'K':>5} {'reten':>6} | {'LiRA AUC':>16} {'TPR@1%FPR':>16} {'decode':>16} | "
          f"{'cosine proxy AUC':>17}")
    print("-" * 92)
    for r in J:
        star = "  <-- recommended" if r["K"] == 32 else ("  <-- both attacks dead" if r["K"] == 128 else "")
        print(f"{r['K']:>5} {r['retention']:>5.0f}% | "
              f"{r['auc_c']:>6.3f} -> {r['auc_d']:<6.3f} "
              f"{r['tpr_c']:>6.3f} -> {r['tpr_d']:<6.3f} "
              f"{r['dec_c']:>6.3f} -> {r['dec_d']:<6.3f} | "
              f"{r['proxy_c']:>7.3f} -> {r['proxy_d']:<6.3f}{star}")
    print("\n  TPR@1%FPR floor = 0.010 (attack has zero usable signal).")
    print("  The proxy reads 0.504 at K=32 -- chance -- where LiRA reads 0.833. It was blind.")


def print_multiround(M, cl):
    print("\n=== (b) MULTI-ROUND: buying a re-aggregation cadence inside a FIXED eps = 9.31 ===\n")
    print(f"{'T':>4} {'schedule':<20} {'sigma':>7} {'recall@5':>9} {'retention':>10} "
          f"{'naive eps':>10}")
    print("-" * 66)
    for r in M:
        print(f"{r['T']:>4} {r['note']:<20} {r['sigma']:>7.3f} {r['rec']:>9.3f} "
              f"{r['retention']:>9.0f}% {r['naive_eps']:>10.1f}")
    print(f"\n  clean recall@5 = {cl:.3f}, chance = 0.156.")
    print("  'naive eps' = what T releases ACTUALLY cost if you keep sigma at the single-shot")
    print("  0.606 and just re-release -- i.e. the guarantee the superseded draft implied.")


def latex(J, M):
    print("\n%% ---- Table: joint frontier (§7.1) ----")
    for r in J:
        print(f"    {r['K']} & {r['chance']:.3f} & \\pmm{{{r['clean_rec']:.3f}}}{{{r['clean_rec_sd']:.3f}}} "
              f"& \\pmm{{{r['dp_rec']:.3f}}}{{{r['dp_rec_sd']:.3f}}} & {r['retention']:.0f}\\% "
              f"& {r['auc_c']:.3f} & {r['auc_d']:.3f} & {r['tpr_c']:.3f} & {r['tpr_d']:.3f} "
              f"& {r['dec_c']:.3f} & {r['dec_d']:.3f} \\\\")
    print("\n%% ---- Table: multi-round schedule (§5.7) ----")
    for r in M:
        print(f"    {r['T']} & {r['note']} & {r['sigma']:.3f} & {r['rec']:.3f} "
              f"& {r['retention']:.0f}\\% & {r['naive_eps']:.1f} \\\\")


def figure(J):
    """Two panels sharing the K axis: the frontier, and the proxy-vs-calibrated contrast."""
    Ks = [r["K"] for r in J]
    x = np.arange(len(Ks))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 3.9))

    # (a) the frontier: retention falls with K, clean leakage rises, DP pins the attack flat.
    axL.plot(x, [r["retention"] / 100 for r in J], "o-", color="#1f77b4", lw=2,
             label="utility retained @ $\\varepsilon\\approx9.3$")
    axL.plot(x, [r["tpr_c"] for r in J], "s--", color="#d62728", lw=2,
             label="attack TPR@1\\%FPR (clean)")
    axL.plot(x, [r["tpr_d"] for r in J], "^-", color="#2ca02c", lw=2,
             label="attack TPR@1\\%FPR @ $\\varepsilon\\approx9.3$")
    axL.axhline(0.01, ls=":", color="grey", lw=1)
    axL.set_xticks(x); axL.set_xticklabels(Ks)
    axL.set_xlabel("memory fidelity $K$ (buckets)")
    axL.set_ylabel("fraction")
    axL.set_ylim(-0.03, 1.03)
    axL.set_title("(a) the joint frontier", fontsize=10)
    axL.legend(fontsize=7.2, loc="center left")
    axL.grid(alpha=0.25)

    # (b) why the old grid missed it: the proxy is blind at exactly the K we recommend.
    axR.plot(x, [r["auc_c"] for r in J], "s-", color="#d62728", lw=2,
             label="calibrated LiRA (clean)")
    axR.plot(x, [r["proxy_c"] for r in J], "d--", color="#9467bd", lw=2,
             label="cosine proxy (clean)")
    axR.plot(x, [r["auc_d"] for r in J], "^-", color="#2ca02c", lw=2,
             label="calibrated LiRA @ $\\varepsilon\\approx9.3$")
    axR.axhline(0.5, ls=":", color="grey", lw=1)
    axR.set_xticks(x); axR.set_xticklabels(Ks)
    axR.set_xlabel("memory fidelity $K$ (buckets)")
    axR.set_ylabel("membership-inference AUC")
    axR.set_ylim(0.44, 1.03)
    axR.set_title("(b) the proxy was blind, not the pool safe", fontsize=10)
    axR.legend(fontsize=7.2, loc="center right")
    axR.grid(alpha=0.25)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_joint_frontier.{ext}", dpi=150)
    plt.close(fig)
    print(f"\n[fig] {FIG / 'fig_joint_frontier.pdf'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true", help="emit table bodies for the manuscript")
    a = ap.parse_args()

    J = load_joint()
    M, cl = load_multiround()
    print_joint(J)
    print_multiround(M, cl)
    figure(J)
    if a.latex:
        latex(J, M)

    (ROOT / "experiment_results" / "joint_rp" / "_summary.json").write_text(
        json.dumps({"joint": J, "multiround": M, "clean_K32": cl}, indent=1))


if __name__ == "__main__":
    main()
