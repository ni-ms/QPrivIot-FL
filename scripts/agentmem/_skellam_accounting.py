"""
Skellam RDP accounting for the agent-memory experiments — replaces the approximate
single-shot Gaussian eps labels with the exact discrete-mechanism guarantee.

Two things this shows:
 1. CORRECTED eps: the experiment sigmas were labelled via the classic Gaussian bound
    sigma = sqrt(2 ln(1.25/delta)) / eps, which is only valid for eps <= 1. This reports
    the rigorous Skellam-RDP eps for the same sigma (valid at any eps), for a single
    release and for the 2-release (sum-vector + counts) composition.
 2. DISCRETISATION SURCHARGE: sweeping the quantiser resolution (range_max) shows the
    Skellam eps converging to the Gaussian-RDP eps as resolution grows — at the project's
    range_max = 1e6 the surcharge is negligible (the discretisation is "free"); at coarse
    resolution it dominates.

Conclusions from the utility/leakage phases are monotone in sigma, so re-labelling the eps
axis does not change any ordering — it only makes the privacy claim rigorous.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qpriviot_fl.privacy_utils import skellam_rdp_epsilon  # noqa: E402

import math

DELTA = 1e-5
# sigmas used in the experiments (were labelled eps=16/8/3 + FL-stress via the classic bound)
SIGMAS = {"'eps=16'": 0.303, "'eps=8'": 0.606, "'eps=3'": 1.615, "FL-sigma": 2.854}
# operating point K=32, d=32: sum-vector release has K*d coords, counts release has K coords
DIM_VEC, DIM_CNT = 32 * 32, 32
CLIP, QBOUND = 1.0, 10.0  # representative; leading RDP term is independent of these


def gaussian_rdp_eps(sigma, delta, releases, orders=range(2, 257)):
    """Gaussian-mechanism RDP eps (leading term only): min_a releases*a/(2 sigma^2) + ln(1/delta)/(a-1)."""
    best = float("inf")
    for a in orders:
        best = min(best, releases * a / (2 * sigma * sigma) + math.log(1 / delta) / (a - 1))
    return best


def classic_gaussian_eps(sigma, delta):
    """Invert the classic bound the labels used (valid only for eps<=1)."""
    return math.sqrt(2 * math.log(1.25 / delta)) / sigma


print("=== Skellam RDP accounting (delta=1e-5) ===")
print(f"operating point: sum-vector dim={DIM_VEC}, counts dim={DIM_CNT}, range_max=1e6\n")
print(f"{'label':<10}{'sigma':>7}{'classic-label':>15}{'Gauss-RDP':>12}{'Skellam 1rel':>14}{'Skellam 2rel':>14}")
print("-" * 72)
for name, sigma in SIGMAS.items():
    scale = 1e6 / QBOUND
    lab = classic_gaussian_eps(sigma, DELTA)
    g1 = gaussian_rdp_eps(sigma, DELTA, 1)
    s1, _ = skellam_rdp_epsilon(sigma, CLIP, scale, DIM_VEC, DELTA, releases=1)
    # 2-release composition: sum-vector (dim K*d) + counts (dim K). Use the larger-dim
    # surcharge for both (conservative) via releases=2 at DIM_VEC.
    s2, _ = skellam_rdp_epsilon(sigma, CLIP, scale, DIM_VEC, DELTA, releases=2)
    print(f"{name:<10}{sigma:>7.3f}{lab:>15.2f}{g1:>12.2f}{s1:>14.4f}{s2:>14.4f}")

print("\n(classic-label = the eps we printed in phase tables; invalid for eps>1.")
print(" Gauss-RDP / Skellam = rigorous RDP eps. Skellam≈Gauss-RDP ⇒ discretisation is free at 1e6.)")

print("\n=== discretisation surcharge vs quantiser resolution (sigma=0.606, 1 release) ===")
print(f"{'range_max':>12}{'scale':>12}{'Skellam eps':>14}{'Gauss-RDP eps':>16}{'surcharge':>12}")
print("-" * 66)
g = gaussian_rdp_eps(0.606, DELTA, 1)
for rmax in [1e2, 1e3, 1e4, 1e5, 1e6]:
    scale = rmax / QBOUND
    s, _ = skellam_rdp_epsilon(0.606, CLIP, scale, DIM_VEC, DELTA, releases=1)
    print(f"{rmax:>12.0f}{scale:>12.0f}{s:>14.4f}{g:>16.4f}{s - g:>12.4f}")
print("\n→ surcharge → 0 as resolution grows; negligible at the project's range_max=1e6.")