"""Signal sensitivity analysis for adaptive privacy - FIXED."""

import torch
import torch.nn as nn
from typing import Dict, List


def compute_parameter_sensitivity(tensor: torch.Tensor, method: str = "variance") -> float:
    """
    Compute sensitivity score for a parameter tensor.
    Higher sensitivity = more privacy protection needed.
    
    Args:
        tensor: Parameter tensor to analyze
        method: "variance", "std", or "range"
    
    Returns:
        Sensitivity score in [0, 1]
    """
    if tensor.numel() == 0:
        return 0.0

    tensor_float = tensor.float()

    if method == "variance":

        mean = torch.abs(tensor_float).mean().item()
        std = torch.std(tensor_float).item()

        if mean < 1e-8:
            return 0.0

        cv = std / mean
        sensitivity = min(1.0, cv / 2.0)

    elif method == "std":

        std = torch.std(tensor_float).item()
        sensitivity = min(1.0, std * 5.0)

    elif method == "range":

        min_val = tensor_float.min().item()
        max_val = tensor_float.max().item()
        range_val = max_val - min_val
        sensitivity = min(1.0, range_val / 2.0)

    else:
        raise ValueError(f"Unknown method: {method}")

    return sensitivity


def analyze_model_sensitivity(model: nn.Module, method: str = "std") -> Dict[str, float]:
    """
    Analyze sensitivity for each layer/parameter in the model.
    
    Args:
        model: PyTorch model
        method: Sensitivity computation method
    
    Returns:
        Dictionary mapping parameter name to sensitivity score
    """
    sensitivities = {}

    for name, param in model.named_parameters():
        if param.requires_grad and param.data is not None:
            sens = compute_parameter_sensitivity(param.data, method=method)
            sensitivities[name] = sens

    return sensitivities


def compute_gradient_sensitivity(
        model: nn.Module,
        method: str = "std"
) -> Dict[str, float]:
    """
    Compute sensitivity based on current gradients (during training).
    More accurate than parameter-based sensitivity.
    
    Args:
        model: PyTorch model with computed gradients
        method: Sensitivity computation method
    
    Returns:
        Dictionary mapping parameter name to gradient sensitivity
    """
    grad_sensitivities = {}

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            sens = compute_parameter_sensitivity(param.grad, method=method)
            grad_sensitivities[name] = sens
        else:
            grad_sensitivities[name] = 0.0

    return grad_sensitivities


def get_average_sensitivity(sensitivities: Dict[str, float]) -> float:
    """
    Get mean sensitivity across all parameters.
    
    Returns:
        Average sensitivity, or 0.5 as default if no sensitivities
    """
    if not sensitivities:
        return 0.5

    valid_sens = [s for s in sensitivities.values() if s > 0]
    if not valid_sens:
        return 0.5

    return sum(valid_sens) / len(valid_sens)


def get_sensitive_layers(
        sensitivities: Dict[str, float],
        threshold: float = 0.5
) -> List[str]:
    """
    Identify highly sensitive layers needing more privacy protection.
    
    Args:
        sensitivities: Dictionary of sensitivity scores
        threshold: Minimum sensitivity to be considered "sensitive"
    
    Returns:
        List of layer names above threshold
    """
    return [name for name, sens in sensitivities.items() if sens >= threshold]


def adaptive_clipping_norms(
        sensitivities: Dict[str, float],
        base_norm: float = 1.0,
        sensitivity_range: float = 0.5
) -> Dict[str, float]:
    """
    Compute per-layer clipping norms based on sensitivity.
    
    MORE SENSITIVE layers get TIGHTER clipping (lower max_grad_norm)
    to provide more privacy protection.
    
    Args:
        sensitivities: Dictionary of sensitivity scores
        base_norm: Base clipping norm
        sensitivity_range: How much to vary clipping (0.5 = ±50%)
    
    Returns:
        Dictionary mapping layer name to max_grad_norm
    """
    clipping_norms = {}

    for name, sens in sensitivities.items():
        norm = base_norm * (1.0 + sensitivity_range * (1.0 - 2.0 * sens))
        clipping_norms[name] = max(0.1, norm)

    return clipping_norms


def get_layer_wise_noise_multipliers(
        sensitivities: Dict[str, float],
        base_noise: float = 1.0,
        sensitivity_factor: float = 0.5
) -> Dict[str, float]:
    """
    Compute per-layer noise multipliers based on sensitivity.
    
    MORE SENSITIVE layers get MORE noise for additional privacy protection.
    
    Args:
        sensitivities: Dictionary of sensitivity scores
        base_noise: Base noise multiplier
        sensitivity_factor: How much to increase noise (0.5 = up to +50%)
    
    Returns:
        Dictionary mapping layer name to noise_multiplier
    """
    noise_multipliers = {}

    for name, sens in sensitivities.items():
        noise = base_noise * (1.0 + sensitivity_factor * sens)
        noise_multipliers[name] = max(0.1, noise)

    return noise_multipliers


def print_sensitivity_report(sensitivities: Dict[str, float]):
    """Print formatted sensitivity analysis report."""
    if not sensitivities:
        print("No sensitivity data available")
        return

    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS REPORT")
    print("=" * 70)

    sorted_sens = sorted(sensitivities.items(), key=lambda x: -x[1])

    for name, sens in sorted_sens[:10]:
        bar_len = int(sens * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"{name:30s} | {bar} | {sens:.4f}")

    print("=" * 70)
    print(f"Average Sensitivity: {get_average_sensitivity(sensitivities):.4f}")
    print(f"Sensitive Layers (>0.5): {len(get_sensitive_layers(sensitivities))}")
    print("=" * 70 + "\n")


if __name__ == "__main__":

    from qpriviot_fl.task import Net

    model = Net()

    sens = analyze_model_sensitivity(model, method="std")

    print_sensitivity_report(sens)

    clipping_norms = adaptive_clipping_norms(sens)
    print("\n📊 Adaptive Clipping Norms:")
    for name, norm in list(clipping_norms.items())[:5]:
        print(f"  {name:30s}: {norm:.3f}")
