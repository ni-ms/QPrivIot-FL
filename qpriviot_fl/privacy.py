"""Adaptive differential privacy - FIXED for Flower simulation."""

import torch
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from typing import Dict, Optional, Tuple, List


def prepare_model_for_dp(model: torch.nn.Module) -> torch.nn.Module:
    """Make model compatible with Opacus - returns a NEW model instance."""

    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        model = ModuleValidator.fix(model)
    return model


def attach_dp_to_optimizer(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: torch.utils.data.DataLoader,
        noise_multiplier: float,
        max_grad_norm: float,
        device: torch.device
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, torch.utils.data.DataLoader, PrivacyEngine]:
    """
    Wrap model and optimizer with Opacus PrivacyEngine.
    FIXED: Handles Flower simulation environment properly.
    """

    model = prepare_model_for_dp(model)

    optimizer_class = type(optimizer)
    optimizer_state = optimizer.state_dict()
    lr = optimizer.param_groups[0]['lr']

    optimizer = optimizer_class(model.parameters(), lr=lr)

    privacy_engine = PrivacyEngine(secure_mode=False)

    try:
        model, optimizer, dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )
    except Exception as e:
        print(f"Opacus attachment failed: {e}")
        raise

    return model, optimizer, dataloader, privacy_engine


def compute_adaptive_dp_config(
        device_profile: Dict[str, float],
        sensitivities: Dict[str, float],
        convergence_score: float,
        current_round: int,
        total_rounds: int,
        base_noise: float = 1.2,
        min_noise: float = 0.2,
        base_clipping: float = 1.0
) -> Dict[str, float]:
    """Compute adaptive DP configuration."""

    resource_score = device_profile.get("resource_score", 0.5)
    device_factor = 1.0 - resource_score

    avg_sensitivity = sum(sensitivities.values()) / (len(sensitivities) + 1e-6)
    sensitivity_factor = avg_sensitivity

    progress_ratio = current_round / max(total_rounds, 1)
    convergence_factor = convergence_score * progress_ratio
    noise_reduction = 0.7 * convergence_factor

    noise_multiplier = max(
        min_noise,
        base_noise * (0.5 + 0.3 * device_factor + 0.2 * sensitivity_factor) * (1.0 - noise_reduction)
    )

    max_grad_norm = base_clipping * (0.5 + 1.5 * resource_score)

    return {
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": max_grad_norm,
        "device_factor": device_factor,
        "sensitivity_factor": sensitivity_factor,
        "convergence_factor": convergence_factor,
    }


class PrivacyAccountant:
    """Track privacy budget across rounds."""

    def __init__(self, target_epsilon: float = 10.0, target_delta: float = 1e-5):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.epsilons_per_round: List[float] = []
        self.total_epsilon = 0.0

    def add_round(self, epsilon: Optional[float]):
        if epsilon is not None and epsilon > 0:
            self.epsilons_per_round.append(epsilon)
            self.total_epsilon += epsilon

    def get_remaining_budget(self) -> float:
        return max(0.0, self.target_epsilon - self.total_epsilon)

    def is_budget_exceeded(self) -> bool:
        return self.total_epsilon >= self.target_epsilon

    def get_privacy_report(self) -> Dict:
        return {
            "total_epsilon": self.total_epsilon,
            "target_epsilon": self.target_epsilon,
            "remaining_budget": self.get_remaining_budget(),
            "rounds_completed": len(self.epsilons_per_round),
            "avg_epsilon_per_round": sum(self.epsilons_per_round) / max(len(self.epsilons_per_round), 1),
        }


def generate_dp_noise(shape: Tuple[int, ...], stddev: float, device: torch.device) -> torch.Tensor:
    """Classical Gaussian noise (quantum-ready hook)."""
    return torch.randn(shape, device=device) * stddev


USE_QUANTUM_NOISE = False
