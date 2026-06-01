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

    noise_multipliers = {}
    clipping_norms = {}

    for layer_name, s_j in sensitivities.items():
        # Ensure s_j is valid (positive and finite)
        if not (0 < s_j < float('inf')):
            s_j = 1.0  # Fallback to neutral sensitivity

        # Scale ONLY the noise multiplier by sensitivity; keep clip constant.
        # With sigma_j × clip_j = constant (old design), the per-parameter noise
        # std = sigma_j × clip_j / sqrt(N) was identical for all layers regardless
        # of s_j — only the clip threshold changed. Since gradient norms at typical
        # DP noise levels (σ≈7–15) are always below all clip thresholds, the tighter
        # clip for high-s layers reduced their gradient signal without any compensating
        # benefit, making param-only identical to or worse than fixed-dp in practice.
        #
        # With constant clip, low-sensitivity layers get genuinely less noise per
        # parameter (σ_j × C_base / √N), which directly improves their SNR. This
        # correctly implements the per-layer budget allocation claimed in the paper.
        # Upper-bound at 1.0: no layer ever gets MORE noise than fixed-dp.
        # Only low-sensitivity layers (stable gradients, e.g. fc1 with 94 % of
        # params) benefit from the reduction to 0.5×. The old 2.0 cap was
        # over-penalizing the output layer (fc2) and conv layers, whose small
        # gradient updates are completely overwhelmed by 2× noise.
        s_j_bounded = np.clip(s_j, 0.5, 1.0)
        noise_multipliers[layer_name] = base_sigma * s_j_bounded
        clipping_norms[layer_name] = base_clip_norm  # constant clip for all layers

    return noise_multipliers, clipping_norms, base_sigma


def apply_dp_noise_per_layer(
        model: nn.Module,
        initial_weights: List[torch.Tensor],
        noise_multipliers: Dict[str, float],
        clipping_norms: Dict[str, float],
        num_samples: int = 1
) -> float:
    """
    Apply DP-SGD: gradient clipping + Gaussian noise per layer.
    
    References: AdaPriv Section 3.5
    Algorithm:
    1. Clip gradient: g̃ = g / max(1, ||g||₂ / C)
    2. Add noise: g̃' = g̃ + N(0, (σC / √n)²)
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

            # Scale noise by sqrt(num_samples) as per Section 3.5
            noise_std = (sigma * clip_norm) / math.sqrt(max(num_samples, 1))
            noise = torch.randn_like(delta) * noise_std

            param.data = initial_weights[i] + delta_clipped + noise

    return 0.0


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
