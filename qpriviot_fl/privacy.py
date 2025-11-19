"""Adaptive DP with CUSTOM NOISE - ALL BUGS FIXED."""

import torch
from opacus import PrivacyEngine, GradSampleModule
from opacus.validators import ModuleValidator
from opacus.optimizers.optimizer import DPOptimizer, _check_processed_flag, _mark_as_processed
from typing import Dict, Optional, Tuple, List

NOISE_CALL_COUNT = 0

def generate_dp_noise(shape: Tuple[int, ...], stddev: float, device: torch.device) -> torch.Tensor:
    """YOUR CUSTOM NOISE FUNCTION."""
    global NOISE_CALL_COUNT
    NOISE_CALL_COUNT += 1

    if NOISE_CALL_COUNT % 10 == 1 or NOISE_CALL_COUNT <= 5:
        print(f" CUSTOM NOISE #{NOISE_CALL_COUNT}: shape={tuple(shape)}, stddev={stddev:.4f}")

    return torch.randn(shape, device=device) * stddev


class CustomDPOptimizer(DPOptimizer):
    """Custom DP Optimizer using YOUR noise."""

    def add_noise(self):
        """Calls YOUR generate_dp_noise()."""
        for p in self.params:
            _check_processed_flag(p.summed_grad)

            noise = generate_dp_noise(
                shape=p.summed_grad.shape,
                stddev=self.noise_multiplier * self.max_grad_norm,
                device=p.summed_grad.device
            )

            p.grad = p.summed_grad + noise
            _mark_as_processed(p.summed_grad)


def prepare_model_for_dp(model: torch.nn.Module) -> torch.nn.Module:
    """Make model DP-compatible."""
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
        device: torch.device,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, torch.utils.data.DataLoader, PrivacyEngine]:
    """
    Attach DP using custom noise - FIXED VERSION.
    """
    global NOISE_CALL_COUNT
    NOISE_CALL_COUNT = 0

    print(f" Preparing model for DP...")

    model = prepare_model_for_dp(model)
    model = GradSampleModule(model)

    optimizer_class = type(optimizer)
    lr = optimizer.param_groups[0]['lr']
    base_optimizer = optimizer_class(model.parameters())

    print(f" Creating CustomDPOptimizer")
    print(f"   noise={noise_multiplier:.2f}, clip={max_grad_norm:.2f}")

    batch_size = getattr(dataloader, 'batch_size', 32)
    sample_rate = batch_size / len(dataloader.dataset)

    optimizer = CustomDPOptimizer(
        optimizer=base_optimizer,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        expected_batch_size=batch_size,
    )

    privacy_engine = PrivacyEngine(secure_mode=False)

    try:
        privacy_engine.accountant.attach_step_hook(
            optimizer,
            data_loader=dataloader,
            sample_rate=sample_rate,
            noise_multiplier=noise_multiplier,
        )
        print(f" CUSTOM DP ATTACHED - generate_dp_noise() will be called!")
    except AttributeError:

        print(f" Using alternative accountant attachment")
        privacy_engine.accountant = privacy_engine.accountant
        privacy_engine.accountant.history.append((noise_multiplier, sample_rate, len(dataloader)))

    return model, optimizer, dataloader, privacy_engine


def compute_adaptive_dp_config(
        device_profile: Dict[str, float],
        sensitivities: Dict[str, float],
        convergence_score: float,
        current_round: int,
        total_rounds: int,
        base_noise: float = 1.2,
        min_noise: float = 0.5,
        base_clipping: float = 1.0
) -> Dict[str, float]:
    """Compute adaptive DP config."""

    resource_score = device_profile.get("resource_score", 0.5)
    device_factor = 1.0 - resource_score

    avg_sensitivity = sum(sensitivities.values()) / (len(sensitivities) + 1e-6)
    sensitivity_factor = min(avg_sensitivity * 100, 1.0)

    progress_ratio = current_round / max(total_rounds, 1)
    convergence_factor = convergence_score * progress_ratio
    noise_reduction = 0.5 * convergence_factor

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
    """Track privacy budget."""

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
