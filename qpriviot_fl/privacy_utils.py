"""
Privacy accounting and adaptive noise allocation for federated learning.

This module implements:
1. Renyi Differential Privacy (RDP) composition [Mironov 2017]
2. Adaptive gradient clipping [Andrew+ 2021]
3. Layer-wise sensitivity analysis [Fu+ 2022]

All formulas are documented with paper citations.
"""

import math
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict


class RenyiPrivacyAccountant:
    """
    Tracks privacy budget using Renyi Differential Privacy composition.
    RDP provides tighter composition than standard (ε,δ)-DP.
    
    References:
    -----------
    [Mironov 2017] "Renyi Differential Privacy"
    [Abadi+ 2016] "Deep Learning with Differential Privacy"
    """

    def __init__(
            self,
            target_epsilon: float = 10.0,
            target_delta: float = 1e-5,
            orders: Optional[List[float]] = None
    ):
        """
        Initialize RDP accountant.
        
        Args:
            target_epsilon: Target privacy parameter
            target_delta: Target failure probability
            orders: Renyi orders to track (default: [1.5, 1.75, ..., 64])
        """
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta

        if orders is None:

            self.orders = [1.0 + x / 10.0 for x in range(5, 50)] + list(range(11, 65))
        else:
            self.orders = orders

        self.rdp_history: List[Dict[float, float]] = []

    def add_round(
            self,
            noise_multiplier: float,
            sampling_rate: float,
            steps: int = 1
    ) -> None:
        """
        Add privacy cost of one round to account.
        
        Uses RDP composition theorem:
        RDP_α(M₁ ∘ M₂) = RDP_α(M₁) + RDP_α(M₂)
        
        Args:
            noise_multiplier: Gaussian noise σ
            sampling_rate: Sampling probability q
            steps: Number of gradient steps (default: 1)
        """
        rdp_round = {}
        for order in self.orders:
            rdp = self._compute_rdp_gaussian(
                order, noise_multiplier, sampling_rate, steps
            )
            rdp_round[order] = rdp

        self.rdp_history.append(rdp_round)

    @staticmethod
    def _compute_rdp_gaussian(
            alpha: float,
            sigma: float,
            q: float,
            steps: int
    ) -> float:
        """
        Compute RDP for Gaussian mechanism with subsampling.
        
        Formula from [Mironov+ 2019]:
        RDP_α ≤ (α * q² * steps) / (2 * σ²)
        
        This uses privacy amplification by subsampling.
        
        Args:
            alpha: Renyi order
            sigma: Noise multiplier
            q: Sampling rate
            steps: Number of steps
        
        Returns:
            RDP at order alpha
        """
        if sigma == 0:
            return float('inf')

        rdp = (alpha * q * q * steps) / (2.0 * sigma * sigma)
        return rdp

    def get_total_epsilon(self) -> float:
        """
        Convert accumulated RDP to (ε,δ)-DP.
        
        Uses conversion formula from [Mironov 2017]:
        ε(δ) = min_α [RDP_α + log(1/δ) / (α - 1)]
        
        Returns:
            Total epsilon spent
        """
        if not self.rdp_history:
            return 0.0

        total_rdp = defaultdict(float)
        for round_rdp in self.rdp_history:
            for order, value in round_rdp.items():
                total_rdp[order] += value

        epsilon_values = []
        for order, rdp in total_rdp.items():
            if order == 1.0:
                continue

            eps = rdp + math.log(1.0 / self.target_delta) / (order - 1.0)
            epsilon_values.append(eps)

        return min(epsilon_values) if epsilon_values else float('inf')

    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget."""
        return max(0.0, self.target_epsilon - self.get_total_epsilon())

    def is_budget_exceeded(self) -> bool:
        """Check if privacy budget exceeded."""
        return self.get_total_epsilon() >= self.target_epsilon


class SensitivityTracker:
    """
    Tracks parameter sensitivity across rounds as per AdaPriv Section 3.3.
    s_j = (1 - alpha) * grad_norm_std_j + alpha * loss_impact_j
    """

    def __init__(self, alpha: float = 0.5, window_size: int = 5):
        self.alpha = alpha
        self.window_size = window_size
        self.history = defaultdict(list)
        self.second_moments = defaultdict(float)

    def update(self, norms: Dict[str, float], grads: Optional[Dict[str, torch.Tensor]] = None):
        """Update historical statistics for sensitivity calculation."""
        for name, norm in norms.items():
            self.history[name].append(norm)
            if len(self.history[name]) > self.window_size:
                self.history[name].pop(0)

        if grads:
            for name, grad in grads.items():
                # loss_impact_j estimated using absolute value of second moment
                moment = torch.mean(grad ** 2).item()
                self.second_moments[name] = 0.9 * self.second_moments[name] + 0.1 * moment

    def get_sensitivities(self) -> Dict[str, float]:
        """Compute sensitivity score s_j for each parameter group.
        
        Uses two independently normalized components to ensure layers
        with different gradient characteristics get different scores:
          s_j = (1 - alpha) * norm_std_j_normalized + alpha * loss_impact_j_normalized
        """
        if not self.history:
            return {}

        # Step 1: Compute raw components per layer
        raw_std = {}
        raw_impact = {}
        for name, norms in self.history.items():
            if len(norms) > 1:
                raw_std[name] = float(np.std(norms))
            else:
                # Not enough history yet — use neutral value so all layers start equal.
                # Using norms[0] here caused large layers (fc1, 262k params) to be
                # flagged high-sensitivity purely from dimensionality, not data sensitivity.
                raw_std[name] = 1.0
            raw_impact[name] = abs(self.second_moments.get(name, 0.0))

        # Step 2: Normalize each component independently to mean 1.0
        # This prevents one component from dominating and ensures
        # even small differences in gradient norms translate to different s_j
        names = list(self.history.keys())
        
        std_values = np.array([raw_std[n] for n in names])
        std_mean = np.mean(std_values)
        if std_mean > 1e-10:
            norm_std = {n: raw_std[n] / std_mean for n in names}
        else:
            norm_std = {n: 1.0 for n in names}

        impact_values = np.array([raw_impact[n] for n in names])
        impact_mean = np.mean(impact_values)
        if impact_mean > 1e-10:
            norm_impact = {n: raw_impact[n] / impact_mean for n in names}
        else:
            norm_impact = {n: 1.0 for n in names}

        # Step 3: Combine with alpha weighting
        sensitivities = {}
        for name in names:
            s_j = (1 - self.alpha) * norm_std[name] + self.alpha * norm_impact[name]
            sensitivities[name] = max(0.1, s_j)

        # Step 4: Final normalization to mean 1.0 (preserves relative differences)
        if sensitivities:
            avg = np.mean(list(sensitivities.values()))
            if avg > 1e-10:
                sensitivities = {k: v / avg for k, v in sensitivities.items()}

        return sensitivities


def allocate_adaptive_noise(
        sensitivities: Dict[str, float],
        target_epsilon: float,
        target_delta: float = 1e-5,
        base_clip_norm: float = 1.0,
        base_sigma: Optional[float] = None,
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """
    Allocate layer-specific noise based on AdaPriv sensitivity s_j.

    High sensitivity s_j -> Lower clipping C_j + Higher noise sigma_j
    This provides "stronger privacy guarantees" for sensitive parameters.

    Args:
        base_sigma: If provided (Opacus-calibrated sigma), use it directly as the
                    global noise scale instead of deriving one from target_epsilon.
                    This ensures the per-layer noise is anchored to the formally
                    calibrated value and the accountant stays consistent.

    References: AdaPriv Section 3.3 & 3.5
    """
    if base_sigma is None or base_sigma <= 0:
        # Fallback: single-shot Gaussian mechanism calibration from epsilon
        base_sigma = math.sqrt(2 * math.log(1.25 / target_delta)) / max(target_epsilon, 1e-6)

    if not sensitivities:
        # Default if no sensitivity data yet - return empty dicts and base_sigma
        return {}, {}, base_sigma

    # ── Budget-conserving per-layer noise allocation ────────────────────────
    # Earlier designs scaled noise by clip(s_j, 0.5, 1.0) — i.e. they could only
    # *reduce* a layer's noise, never raise it. That lowers the layer-mean noise
    # multiplier below base_sigma, which the server's RDP accountant charges as a
    # larger ε (observed: param-only spent 9.56 vs fixed-dp's 9.16 at ε=8). The
    # comparison was therefore NOT iso-privacy: param-only quietly ran at a looser
    # budget and still failed to beat fixed-dp on accuracy.
    #
    # Fix: redistribute a FIXED noise budget across layers instead of spending
    # more of it. We allow factors in [0.5, 1.5] (both directions) and renormalize
    # so the unweighted layer-mean factor is exactly 1.0 — which is precisely the
    # statistic the accountant measures (mean over layers, min over clients). Net
    # effect: the accounted noise multiplier equals base_sigma exactly, so
    # param-only and fixed-dp spend identical ε. The adaptivity is now a pure
    # *reallocation*: stable/dominant layers (low s_j, e.g. fc1 with 94% of params)
    # shed noise onto volatile small layers (high s_j) at no privacy cost.
    clip_lo, clip_hi = 0.5, 1.5
    raw_factors = {}
    for layer_name, s_j in sensitivities.items():
        if not (0 < s_j < float('inf')):
            s_j = 1.0  # neutral fallback
        # High sensitivity → more noise; low sensitivity → less. (Same direction
        # as before, wider band so the budget can be conserved by renormalizing.)
        raw_factors[layer_name] = float(np.clip(s_j, clip_lo, clip_hi))

    mean_factor = float(np.mean(list(raw_factors.values()))) if raw_factors else 1.0
    if mean_factor <= 0:
        mean_factor = 1.0

    noise_multipliers = {}
    clipping_norms = {}
    for layer_name, f_j in raw_factors.items():
        # Renormalize so mean(noise_multipliers) == base_sigma → iso-privacy.
        noise_multipliers[layer_name] = base_sigma * (f_j / mean_factor)
        clipping_norms[layer_name] = base_clip_norm  # constant clip for all layers

    return noise_multipliers, clipping_norms, base_sigma


def apply_dp_noise_per_layer(
        model: nn.Module,
        initial_weights: List[torch.Tensor],
        noise_multipliers: Dict[str, float],
        clipping_norms: Dict[str, float],
        num_samples: int = 1,
        add_noise: bool = True,
) -> float:
    """
    Apply DP-SGD: gradient clipping + Gaussian noise per layer.

    References: AdaPriv Section 3.5
    Algorithm:
    1. Clip gradient: g̃ = g / max(1, ||g||₂ / C)
    2. Add noise: g̃' = g̃ + N(0, (σC / √n)²)

    add_noise=False performs clipping ONLY (no continuous noise). Used by the
    distributed-Skellam SecAgg path, which injects discrete integer-domain noise
    *after* quantization instead (see apply_distributed_skellam_noise).
    """
    with torch.no_grad():
        for i, (name, param) in enumerate(model.named_parameters()):
            if i >= len(initial_weights):
                break

            delta = param.data - initial_weights[i]

            clip_norm = clipping_norms.get(name, 1.0)
            sigma = noise_multipliers.get(name, 0.0)

            norm = torch.norm(delta).item()
            clip_factor = min(1.0, clip_norm / (norm + 1e-8))
            delta_clipped = delta * clip_factor

            if add_noise:
                # Client-level DP: the sensitivity is the clipped CLIENT DELTA (norm C),
                # so the calibrated noise is N(0, (σ·C)²) on the delta directly. The former
                # `/√num_samples` factor (≈√6000≈77×) silently deflated σ to ~no-DP — it
                # conflated per-example DP-SGD averaging with client-level update perturbation
                # and made every legacy fixed-dp result non-private (see PAPER_STATUS FINDING 2).
                noise_std = sigma * clip_norm
                noise = torch.randn_like(delta) * noise_std
                param.data = initial_weights[i] + delta_clipped + noise
            else:
                param.data = initial_weights[i] + delta_clipped

    return 0.0


def skellam_noise(shape: Tuple, variance: float, rng: np.random.Generator) -> np.ndarray:
    """
    Symmetric Skellam noise: Poisson(μ) − Poisson(μ), with variance = 2μ.

    Integer-valued and CLOSED UNDER ADDITION (the sum of independent Skellams is
    itself Skellam), which is exactly the property needed for distributed DP under
    secure aggregation: N clients each add a share, and the masked integer sum is a
    single Skellam mechanism with the target aggregate variance.

    Reference: Agarwal et al., "The Skellam Mechanism for Differentially Private
    Federated Learning" (NeurIPS 2021).
    """
    if variance <= 0:
        return np.zeros(shape, dtype=np.int64)
    mu = variance / 2.0
    a = rng.poisson(mu, size=shape).astype(np.int64)
    b = rng.poisson(mu, size=shape).astype(np.int64)
    return a - b


def skellam_rdp_epsilon(
        sigma: float,
        clip_norm: float,
        scale: float,
        dim: int,
        target_delta: float = 1e-5,
        releases: int = 1,
        l1_sensitivity: Optional[float] = None,
        orders: Optional[List[int]] = None,
        clip_eps: float = 0.0,
        clip_releases: int = 0,
) -> Tuple[float, float]:
    """Exact (ε, δ)-DP for the (distributed) Skellam mechanism via RDP.

    Agarwal, Kairouz & Liu, "The Skellam Mechanism for Differentially Private
    Federated Learning" (NeurIPS 2021), Theorem 3.5. In their parametrisation μ is the
    PER-COORDINATE VARIANCE of the released Skellam noise (each of the two Poissons has
    mean μ/2, so Var = μ; PMF e^{-μ}I_k(μ); E‖noise‖² = d·μ). For integer α > 1 and
    integer ℓ2 sensitivity Δ2 (with ℓ1 sensitivity Δ1):

        ε_RDP(α) ≤ α·Δ2²/(2μ) + min{ ((2α−1)·Δ2² + 6·Δ1)/(4μ²),  3·Δ1/(2μ) }

    The leading term equals the Gaussian mechanism's α·Δ2²/(2μ); the min{…} is the
    discretisation surcharge, which vanishes as μ (≈ resolution²) grows — the bound is at
    most a (1+O(1/μ)) factor worse than Gaussian.

    Mapping from this project's mechanism (distributed Skellam under SecAgg):
        μ  = aggregate per-coordinate variance = (sigma · clip_norm · scale)²
             (per-client variance (σ·C·s)²/N summed over N clients under SecAgg)
        Δ2 = integer ℓ2 sensitivity of one user's clipped, quantised update = clip_norm · scale
        Δ1 = integer ℓ1 sensitivity ≤ √dim · Δ2   (worst case; pass a tighter value if known)
        scale = range_max / quantization_bound

    `releases` composes that many identical Skellam releases additively in RDP (e.g. a
    separately-noised sum-vector and count vector) before the (ε, δ) conversion.

    Returns (epsilon, best_order).
    """
    if sigma <= 0:
        return float("inf"), 0.0
    if orders is None:
        orders = list(range(2, 257))

    mu = (sigma * clip_norm * scale) ** 2
    delta2 = clip_norm * scale
    delta1 = l1_sensitivity if l1_sensitivity is not None else math.sqrt(dim) * delta2
    d2sq = delta2 * delta2

    # Cost of DP-selecting the clip bound C (exponential mechanism, eps_c-DP each time):
    # eps_c-DP  =>  (eps_c^2/2)-zCDP  =>  RDP_alpha <= alpha * eps_c^2 / 2  [Bun & Steinke 2016].
    clip_rho = 0.0
    if clip_eps and clip_eps > 0 and clip_releases > 0:
        clip_rho = clip_releases * (clip_eps ** 2) / 2.0

    best_eps = float("inf")
    best_alpha = 0.0
    for alpha in orders:
        rdp_lead = alpha * d2sq / (2.0 * mu)
        corr = min(((2 * alpha - 1) * d2sq + 6 * delta1) / (4 * mu * mu),
                   3 * delta1 / (2 * mu))
        rdp = releases * (rdp_lead + corr) + alpha * clip_rho
        eps = rdp + math.log(1.0 / target_delta) / (alpha - 1)
        if eps < best_eps:
            best_eps = eps
            best_alpha = float(alpha)
    return best_eps, best_alpha


# Public geometric grid of candidate clip bounds. Each note embedding is L2-unit, and a user with
# n_u notes has ||V_u||_2 in [sqrt(n_u), n_u]; for agent-memory payloads of tens of notes per user
# that brackets [2, 64]. The grid is COARSE on purpose: the exponential mechanism's rank error grows
# like log|grid|/eps_c, and end-to-end utility is flat in C over a factor ~2 (see paper 5.2), so a
# fine grid buys nothing and costs accuracy.
PUBLIC_CLIP_GRID = (2.0, 64.0, 16)


def dp_quantile_clip(norms, q, eps_c, rng, grid=PUBLIC_CLIP_GRID):
    """Differentially-private estimate of the q-quantile of per-user payload norms.

    The clip bound C is a PUBLIC parameter of the released mechanism, yet the natural choice
    (the empirical 95th percentile of user norms) is a function of the private corpus. Releasing
    it is an unaccounted release. We instead select C from a public candidate grid with the
    EXPONENTIAL MECHANISM (Smith 2011; McSherry & Talwar 2007) using the rank utility

        u(c) = - | #{u : ||V_u||_2 <= c}  -  q*N |

    Neighbouring = ADD/REMOVE one user (unbounded DP), the same relation that makes the Skellam
    sensitivity Delta_2 = C exact. Both the count and the target q*N move when a user leaves:
    removing a user with norm <= c shifts the argument of |.| by 1 - q = 0.05; removing one with
    norm > c shifts it by q. So Delta_u = max(1 - q, q) = q = 0.95 <= 1, and sampling with

        Pr[C = c]  proportional to  exp( eps_c * u(c) / 2 )

    is (q * eps_c)-DP. We charge the full eps_c, a ~5% conservative margin.

    eps_c composes with the Skellam release in RDP via pure-DP -> zCDP:
    an eps_c-DP mechanism is (eps_c^2 / 2)-zCDP, hence RDP_alpha <= alpha * eps_c^2 / 2
    (Bun & Steinke 2016). See `skellam_rdp_epsilon(..., clip_eps=...)`.

    eps_c <= 0 disables the mechanism and returns the (non-private) empirical quantile, which is
    only for reproducing the superseded, unaccounted baseline.
    """
    norms = np.asarray(norms, dtype=np.float64)
    if eps_c is None or eps_c <= 0:
        return float(np.percentile(norms, 100.0 * q))
    lo, hi, n = grid
    cand = np.geomspace(lo, hi, n)
    counts = np.searchsorted(np.sort(norms), cand, side="right")
    util = -np.abs(counts - q * len(norms))          # Delta_u = 1 (replace-one)
    logits = (eps_c / 2.0) * util
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    return float(rng.choice(cand, p=p))


def apply_distributed_skellam_noise(
        quantized_deltas: List[np.ndarray],
        sigma: float,
        clip_norm: float,
        num_clients: int,
        quantization_bound: float,
        range_max: int,
        distributed: bool,
        seed: int,
) -> List[np.ndarray]:
    """
    Add per-client Skellam DP noise in the integer (quantized) domain, before masking.

    The quantizer maps a float in [−B, B] to an integer via scale s = range_max / B,
    so a per-coordinate float sensitivity·σ of (σ·C) becomes (σ·C·s) in the integer
    domain. The Gaussian-mechanism-equivalent target is per-coordinate variance
    (σ·C·s)² on the released quantity.

      • distributed=True  : per-client variance = (σ·C·s)² / N.
                            SecAgg sums N clients → aggregate variance (σ·C·s)²
                            (central-DP-equivalent noise on the SUM; mean noise ≈ σC/N).
                            Requires SecAgg to hide individual contributions.
      • distributed=False : per-client variance = (σ·C·s)²  ("local DP": every client
                            is itself σ-DP without any trust assumption; mean noise ≈ σC/√N).

    Both report the same noise multiplier σ to the accountant, so the *claimed* ε is
    identical — the distributed variant simply achieves it with ~√N less aggregate noise.
    """
    rng = np.random.default_rng(seed)
    scale = range_max / max(quantization_bound, 1e-9)
    base_var = (sigma * clip_norm * scale) ** 2
    per_client_var = base_var / max(num_clients, 1) if distributed else base_var

    noised = []
    for arr in quantized_deltas:
        noise = skellam_noise(arr.shape, per_client_var, rng)
        noised.append((arr.astype(np.int64) + noise).astype(np.int64))
    return noised


def compute_adaptive_clipping_norm(
        previous_norm: float,
        observed_norms: List[float],
        quantile: float = 0.5,
        learning_rate: float = 0.2,
        min_norm: float = 0.01,
        max_norm: float = 10.0
) -> float:
    """
    Adaptive clipping using quantile-based update.
    
    Algorithm from [Andrew+ 2021]:
    C_{t+1} = C_t * (1 - lr) + quantile(||g||) * lr
    
    Intuition: Track median gradient norm to avoid clipping too much or too little.
    
    Args:
        previous_norm: Clipping norm from previous round
        observed_norms: Gradient norms from current round
        quantile: Target quantile (default: 0.5 = median)
        learning_rate: EMA learning rate
        min_norm: Minimum clipping norm
        max_norm: Maximum clipping norm
    
    Returns:
        Updated clipping norm
    """
    if not observed_norms:
        return previous_norm

    target_norm = float(np.quantile(observed_norms, quantile))

    new_norm = previous_norm * (1.0 - learning_rate) + target_norm * learning_rate

    new_norm = np.clip(new_norm, min_norm, max_norm)

    return float(new_norm)


def quantize(params: List[np.ndarray], clip_range: float = 1.0, range_max: int = 1_000_000) -> List[np.ndarray]:
    """Convert Float weights to Integers for Secure Aggregation (Quantization)."""
    quantized_list = []
    for arr in params:
        clipped = np.clip(arr, -clip_range, clip_range)
        scaled = (clipped / clip_range) * range_max
        quantized_list.append(np.round(scaled).astype(np.int64))
    return quantized_list


def dequantize(params: List[np.ndarray], clip_range: float = 1.0, range_max: int = 1_000_000) -> List[np.ndarray]:
    """Convert Integers back to Floats after aggregation (Dequantization)."""
    dequantized_list = []
    for arr in params:
        arr_float = arr.astype(np.float64)
        unscaled = (arr_float / range_max) * clip_range
        dequantized_list.append(unscaled.astype(np.float32))
    return dequantized_list


def generate_zero_sum_masks(shapes: List[Tuple], num_clients: int, seed: int) -> List[List[np.ndarray]]:
    """
    Generates 'num_clients' lists of masks using a seed. The sum of all masks is guaranteed to be ZERO.
    """
    rng = np.random.default_rng(seed)
    masks_per_client = [[] for _ in range(num_clients)]
    
    # Bound the max range relative to num_clients to prevent int64 overflow
    # For int64 max ~9e18. If num_clients is large, we reduce the range limit
    # Max sum possible is num_clients * range_limit.
    base_range = 2_000_000
    safe_limit = int((9e18 / max(1, num_clients)) ** 0.5) # somewhat arbitrary conservative bound
    range_limit = min(base_range, safe_limit)

    for shape in shapes:

        random_masks = [rng.integers(-range_limit, range_limit, size=shape, dtype=np.int64)
                        for _ in range(num_clients - 1)]

        sum_others = sum(random_masks)
        last_mask = -sum_others

        for i in range(num_clients - 1):
            masks_per_client[i].append(random_masks[i])
        masks_per_client[num_clients - 1].append(last_mask)

    return masks_per_client
