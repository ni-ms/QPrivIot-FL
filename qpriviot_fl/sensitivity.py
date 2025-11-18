"""
Per-Layer Sensitivity Analysis for Adaptive Differential Privacy.

[1] Dwork & Roth (2014). "The Algorithmic Foundations of Differential Privacy"
    Foundations and Trends in TCS, 9(3-4), 211-407. doi:10.1561/0400000042
[2] Abadi et al. (2016). "Deep Learning with Differential Privacy"
    ACM CCS. doi:10.1145/2976749.2978318
[3] Mironov (2017). "Rényi Differential Privacy"
    IEEE CSF. doi:10.1109/CSF.2017.11
[4] Li et al. (2020). "Multi-site fMRI Analysis Using Privacy-preserving FL"
    Medical Image Analysis, 65, 101765. doi:10.1016/j.media.2020.101765
"""

import logging
import math
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GradientSensitivityAnalyzer:
    def __init__(self, model: nn.Module):
        self.model = model
        self.sensitivity_cache = {}

    def compute_per_sample_gradient_variance(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        criterion: nn.Module = None
    ) -> Dict[str, float]:
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        if isinstance(batch, dict):
            x, y = batch["img"], batch["label"]
        else:
            x, y = batch

        device = next(self.model.parameters()).device
        x, y = x.to(device), y.to(device)

        batch_size = x.size(0)
        if batch_size < 2:
            return {name: 0.0 for name, _ in self.model.named_parameters()}

        per_sample_grads = defaultdict(list)

        for i in range(batch_size):
            self.model.zero_grad()
            xi, yi = x[i:i+1], y[i:i+1]
            output = self.model(xi)
            loss = criterion(output, yi)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    per_sample_grads[name].append(param.grad.detach().clone())

        sensitivities = {}
        for name, grads in per_sample_grads.items():
            if len(grads) > 1:
                stacked_grads = torch.stack(grads)
                norms = torch.norm(stacked_grads.view(stacked_grads.size(0), -1), dim=1)
                sensitivities[name] = torch.std(norms).item()
            else:
                sensitivities[name] = 0.0

        return sensitivities

    def sample_sensitivity_multi_batch(
        self,
        dataloader,
        num_batches: int = 5,
        criterion: nn.Module = None
    ) -> Dict[str, float]:
        all_sensitivities = defaultdict(list)

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            batch_sens = self.compute_per_sample_gradient_variance(batch, criterion)
            for name, sens in batch_sens.items():
                all_sensitivities[name].append(sens)

        avg_sensitivities = {}
        for name, sens_list in all_sensitivities.items():
            avg_sensitivities[name] = np.mean(sens_list) if sens_list else 0.0

        return avg_sensitivities

    def normalize_sensitivities(
        self,
        sensitivities: Dict[str, float]
    ) -> Dict[str, float]:
        if not sensitivities:
            return {}

        sens_values = [s for s in sensitivities.values() if s > 0]
        if not sens_values:
            return {name: 0.5 for name in sensitivities}

        max_sens = max(sens_values)
        min_sens = min(sens_values)

        if max_sens == min_sens:
            return {name: 0.5 for name in sensitivities}

        normalized = {}
        for name, sens in sensitivities.items():
            norm_sens = (sens - min_sens) / (max_sens - min_sens)
            normalized[name] = np.clip(norm_sens, 0.0, 1.0)

        return normalized


class AdaptiveNoiseAllocator:
    @staticmethod
    def allocate_noise_proportional_to_sensitivity(
        sensitivities: Dict[str, float],
        target_epsilon: float,
        delta: float = 1e-5,
        clip_norm: float = 1.0
    ) -> Dict[str, float]:
        """
        Gaussian mechanism [1,2]: ε = C·√(2·ln(1.25/δ)) / σ
        where C = clip_norm (max gradient norm)

        For adaptive allocation:
        σ_i = sensitivity_i · C · √(2·ln(1.25/δ)) · N / ε_total
        """
        privacy_constant = math.sqrt(2 * math.log(1.25 / delta))
        num_layers = len(sensitivities)

        total_weighted_sens = sum(sensitivities.values())
        if total_weighted_sens == 0:
            return {name: clip_norm for name in sensitivities}

        noise_multipliers = {}
        for name, sens in sensitivities.items():
            weight = sens / total_weighted_sens
            sigma = weight * clip_norm * privacy_constant * num_layers / target_epsilon
            noise_multipliers[name] = max(sigma, 0.1)

        return noise_multipliers

    @staticmethod
    def allocate_noise_simple_proportional(
        sensitivities: Dict[str, float],
        base_noise: float = 1.0
    ) -> Dict[str, float]:
        """
        Simplified: σ_i = base_noise · (1 + sensitivity_i)
        Higher sensitivity → more noise [2,4]
        """
        noise_multipliers = {}
        for name, sens in sensitivities.items():
            noise = base_noise * (1.0 + sens)
            noise_multipliers[name] = np.clip(noise, 0.5, 3.0)

        return noise_multipliers


class PrivacyAccountant:
    @staticmethod
    def compute_epsilon_per_layer(
        sensitivity: float,
        noise_multiplier: float,
        delta: float = 1e-5,
        clip_norm: float = 1.0
    ) -> float:
        """
        Analytical Gaussian mechanism [1]:
        ε = C · √(2·ln(1.25/δ)) / σ
        """
        if noise_multiplier <= 0:
            return float('inf')

        privacy_constant = math.sqrt(2 * math.log(1.25 / delta))
        epsilon = (clip_norm * privacy_constant) / noise_multiplier
        return epsilon

    @staticmethod
    def compute_total_epsilon_advanced_composition(
        per_layer_epsilons: List[float],
        delta: float = 1e-5,
        num_rounds: int = 1
    ) -> float:
        """
        Advanced composition [1, Theorem 3.20]:
        ε_total = √(2·k·ln(1/δ')) · ε + k·ε·(e^ε - 1)
        Approximation for small ε: ε_total ≈ ε·√(2·k·ln(1/δ))
        """
        if not per_layer_epsilons:
            return 0.0

        k = len(per_layer_epsilons) * num_rounds
        avg_epsilon = np.mean(per_layer_epsilons)

        if avg_epsilon < 0.1:
            epsilon_total = avg_epsilon * math.sqrt(2 * k * math.log(1/delta))
        else:
            epsilon_total = math.sqrt(2 * k * math.log(1/delta)) * avg_epsilon + \
                           k * avg_epsilon * (math.exp(avg_epsilon) - 1)

        return epsilon_total


def compute_adaptive_sensitivity(
    model: nn.Module,
    dataloader,
    num_batches: int = 5
) -> Tuple[Dict[str, float], Dict[str, float]]:
    analyzer = GradientSensitivityAnalyzer(model)

    raw_sensitivities = analyzer.sample_sensitivity_multi_batch(
        dataloader, num_batches
    )

    normalized_sensitivities = analyzer.normalize_sensitivities(raw_sensitivities)

    return normalized_sensitivities, raw_sensitivities


def get_adaptive_dp_noise_config(
    sensitivities: Dict[str, float],
    target_epsilon: float = 1.0,
    delta: float = 1e-5,
    clip_norm: float = 1.0,
    strategy: str = "proportional"
) -> Dict:
    allocator = AdaptiveNoiseAllocator()

    if strategy == "proportional":
        noise_multipliers = allocator.allocate_noise_simple_proportional(
            sensitivities, base_noise=target_epsilon
        )
    elif strategy == "gaussian":
        noise_multipliers = allocator.allocate_noise_proportional_to_sensitivity(
            sensitivities, target_epsilon, delta, clip_norm
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    accountant = PrivacyAccountant()
    per_layer_epsilons = [
        accountant.compute_epsilon_per_layer(sens, noise_multipliers[name], delta, clip_norm)
        for name, sens in sensitivities.items()
    ]

    total_epsilon = accountant.compute_total_epsilon_advanced_composition(
        per_layer_epsilons, delta, num_rounds=1
    )
    clipping_norms = {name: clip_norm for name in sensitivities}

    return {
        "sensitivities": sensitivities,
        "noise_multipliers": noise_multipliers,
        "clipping_norms": clipping_norms,
        "per_layer_epsilons": per_layer_epsilons,
        "total_epsilon": total_epsilon,
        "strategy": strategy,
        "clip_norm": clip_norm,
        "per_layer": True,
    }


