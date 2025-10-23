"""Signal sensitivity analysis for adaptive privacy."""

import torch
from typing import Dict, List


def compute_parameter_sensitivity(tensor: torch.Tensor) -> float:
    """
    Compute sensitivity score for a parameter tensor.
    Higher variance/instability = higher sensitivity.

    Returns: sensitivity score in [0, 1]
    """
    if tensor.numel() == 0:
        return 0.0

    variance = torch.var(tensor.float()).item()
    std = torch.std(tensor.float()).item()
    mean_abs = torch.abs(tensor.float()).mean().item()

    sensitivity = min(1.0, variance / (1.0 + variance))

    return sensitivity


def analyze_model_sensitivity(model: torch.nn.Module) -> Dict[str, float]:
    """
    Analyze sensitivity for each layer/parameter in the model.

    Returns: dict mapping parameter name to sensitivity score
    """
    sensitivities = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            sens = compute_parameter_sensitivity(param.data)
            sensitivities[name] = sens

    return sensitivities


def compute_gradient_sensitivity(gradients: List[torch.Tensor]) -> Dict[int, float]:
    """
    Compute sensitivity of gradients (useful during training).

    Returns: dict mapping layer index to gradient sensitivity
    """
    grad_sensitivities = {}

    for idx, grad in enumerate(gradients):
        if grad is not None:
            sens = compute_parameter_sensitivity(grad)
            grad_sensitivities[idx] = sens

    return grad_sensitivities


def get_average_sensitivity(sensitivities: Dict[str, float]) -> float:
    """Get mean sensitivity across all parameters."""
    if not sensitivities:
        return 0.5
    return sum(sensitivities.values()) / len(sensitivities)


def get_sensitive_layers(sensitivities: Dict[str, float],
                         threshold: float = 0.7) -> List[str]:
    """
    Identify highly sensitive layers that need more privacy protection.
    """
    return [name for name, sens in sensitivities.items() if sens > threshold]


def adaptive_clipping_norms(sensitivities: Dict[str, float],
                            base_norm: float = 1.0,
                            sensitivity_factor: float = 2.0) -> Dict[str, float]:
    """
    Compute per-layer clipping norms based on sensitivity.
    More sensitive layers get tighter clipping.

    Returns: dict mapping layer name to max_grad_norm
    """
    clipping_norms = {}

    for name, sens in sensitivities.items():
        norm = base_norm * (2.0 - sens * sensitivity_factor)
        clipping_norms[name] = max(0.1, norm)

    return clipping_norms
