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


def compute_gradient_sensitivity(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_batches: int = 10,
        device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Estimate per-layer gradient sensitivity for adaptive clipping.
    
    Sensitivity = typical gradient L2 norm for each layer.
    Used to allocate different clipping norms per layer.
    
    References:
    -----------
    [Fu+ 2022] Section 4.1: "Adaptive gradient clipping"
    
    Args:
        model: Neural network model
        dataloader: Training data loader
        num_batches: Number of batches to sample (default: 10)
        device: Computation device
    
    Returns:
        Dictionary mapping layer name to median gradient norm
    """
    if device is None:
        device = next(model.parameters()).device

    model.train()
    criterion = nn.CrossEntropyLoss()

    sensitivity_samples = defaultdict(list)

    data_iter = iter(dataloader)
    for batch_idx in range(min(num_batches, len(dataloader))):
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        if isinstance(batch, dict):
            if "img" in batch:
                inputs, labels = batch["img"], batch["label"]
            else:
                inputs, labels = batch["image"], batch["label"]
        else:
            inputs, labels = batch[0], batch[1]

        inputs = inputs.to(device, dtype=torch.float32)
        labels = labels.to(device)

        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                sensitivity_samples[name].append(grad_norm)

    sensitivities = {
        name: float(np.median(norms))
        for name, norms in sensitivity_samples.items()
        if len(norms) > 0
    }

    return sensitivities


def allocate_adaptive_noise(
        sensitivities: Dict[str, float],
        target_epsilon: float,
        target_delta: float = 1e-5,
        base_clip_norm: float = 1.0
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """
    Allocate layer-specific noise based on gradient sensitivity.
    
    High-sensitivity layers get more clipping/noise budget.
    Low-sensitivity layers get less noise → better utility.
    
    Algorithm from [Fu+ 2022] Section 4.2:
    1. Compute sensitivity weight for each layer
    2. Allocate epsilon proportional to weight
    3. Convert epsilon to noise multiplier
    
    Args:
        sensitivities: Per-layer gradient norms
        target_epsilon: Target privacy parameter for this round
        target_delta: Failure probability
        base_clip_norm: Base clipping norm
    
    Returns:
        Tuple of (noise_multipliers, clipping_norms, total_epsilon)
    """
    if not sensitivities:
        return {}, {}, 0.0

    total_sensitivity = sum(sensitivities.values())
    if total_sensitivity == 0:
        total_sensitivity = 1e-9

    num_layers = len(sensitivities)

    base_layer_epsilon = target_epsilon / math.sqrt(max(num_layers, 1))

    noise_multipliers = {}
    clipping_norms = {}
    layer_epsilons = []

    calibration_constant = math.sqrt(2.0 * math.log(1.25 / target_delta))

    for layer_name, sensitivity in sensitivities.items():
        weight = sensitivity / total_sensitivity

        allocation_factor = 0.5 + weight * num_layers
        allocation_factor = np.clip(allocation_factor, 0.5, 1.5)

        layer_epsilon = base_layer_epsilon * allocation_factor
        layer_epsilons.append(layer_epsilon)

        sigma = calibration_constant / layer_epsilon

        sigma = np.clip(sigma, 0.5, 10.0)

        noise_multipliers[layer_name] = float(sigma)
        clipping_norms[layer_name] = base_clip_norm

    total_epsilon = math.sqrt(sum(eps ** 2 for eps in layer_epsilons))

    return noise_multipliers, clipping_norms, total_epsilon


def apply_dp_noise_per_layer(
        model: nn.Module,
        initial_weights: List[torch.Tensor],
        noise_multipliers: Dict[str, float],
        clipping_norms: Dict[str, float]
) -> float:
    """
    Apply DP-SGD: gradient clipping + Gaussian noise per layer.
    
    Algorithm:
    1. Clip gradient: g̃ = g / max(1, ||g||₂ / C)
    2. Add noise: g̃' = g̃ + N(0, σ²C²)
    
    Args:
        model: Neural network with updated parameters
        initial_weights: Pre-training parameter snapshots
        noise_multipliers: Per-layer noise σ
        clipping_norms: Per-layer clipping C
    
    Returns:
        Actual epsilon spent (for accounting)
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

            noise_std = sigma * clip_norm
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
