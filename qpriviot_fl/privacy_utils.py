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
        """Compute sensitivity score s_j for each parameter group."""
        sensitivities = {}
        for name, norms in self.history.items():
            grad_norm_std = np.std(norms) if len(norms) > 1 else 0.1
            loss_impact = abs(self.second_moments[name])
            
            # Normalize and combine
            s_j = (1 - self.alpha) * grad_norm_std + self.alpha * loss_impact
            sensitivities[name] = max(0.1, s_j)
            
        # Normalize sensitivities to mean 1.0 to keep clipping norms stable
        if sensitivities:
            avg = np.mean(list(sensitivities.values()))
            sensitivities = {k: v / (avg + 1e-8) for k, v in sensitivities.items()}
            
        return sensitivities


def allocate_adaptive_noise(
        sensitivities: Dict[str, float],
        target_epsilon: float,
        target_delta: float = 1e-5,
        base_clip_norm: float = 1.0
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """
    Allocate layer-specific noise based on AdaPriv sensitivity s_j.
    
    C_j = C_base * s_j
    
    References: AdaPriv Section 3.3 & 3.5
    """
    if not sensitivities:
        # Default if no sensitivity data yet
        return {}, {}, target_epsilon

    noise_multipliers = {}
    clipping_norms = {}
    
    # Calibration constant for Gaussian mechanism
    calibration_constant = math.sqrt(2.0 * math.log(1.25 / target_delta))
    # Standard sigma = calibration / epsilon
    base_sigma = calibration_constant / (target_epsilon + 1e-8)

    for layer_name, s_j in sensitivities.items():
        # High sensitivity s_j -> Lower clipping C_j (Aggressive) or Higher Clipping?
        # Paper says: "Parameters with high s_j receive proportionally more privacy protection (lower clipping and higher noise)"
        # Wait, if s_j is high, and C_j = C_base * s_j, then C_j is HIGHER.
        # But if we want MORE protection, we should CLIP MORE (Lower C_j) or ADD MORE NOISE.
        # Paper says: "Parameter sensitivity analysis that identifies which model parameters require stronger privacy guarantees"
        # "Parameters with high s_j receive proportionally more privacy protection (lower clipping and higher noise)"
        # So C_j should be inversely proportional to s_j if we want "lower clipping" for high s_j?
        # Or does "lower clipping" mean a lower threshold value? Usually "clipping heavily" means a lower threshold.
        # Let's follow the literal "lower clipping" means smaller C_j.
        
        clip_j = base_clip_norm / (s_j + 1e-8) 
        
        # Noise std = sigma * C_j. If we want HIGHER noise for high s_j:
        # sigma_j = base_sigma * s_j
        
        noise_multipliers[layer_name] = base_sigma * s_j
        clipping_norms[layer_name] = clip_j

    return noise_multipliers, clipping_norms, target_epsilon


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
    np.random.seed(seed)
    masks_per_client = [[] for _ in range(num_clients)]
    range_limit = 2_000_000

    for shape in shapes:

        random_masks = [np.random.randint(-range_limit, range_limit, size=shape, dtype=np.int64)
                        for _ in range(num_clients - 1)]

        sum_others = sum(random_masks)
        last_mask = -sum_others

        for i in range(num_clients - 1):
            masks_per_client[i].append(random_masks[i])
        masks_per_client[num_clients - 1].append(last_mask)

    return masks_per_client
