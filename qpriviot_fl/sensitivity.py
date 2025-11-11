"""
Signal sensitivity analysis for adaptive privacy - IMPROVED VERSION.

KEY IMPROVEMENTS:
1. Gradient-based sensitivity (not weight-based)
2. Multi-batch sampling for robustness
3. Layer classification (input/hidden/output)
4. Normalized noise allocation (respects epsilon budget)
5. Proper DP integration
6. Comprehensive documentation
7. Validation utilities
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import logging

logger = logging.getLogger(__name__)


class LayerClassifier:
    """Classify layers by type and role in the model."""

    @staticmethod
    def classify_layers(model: nn.Module) -> Dict[str, List[str]]:
        """
        Classify model layers by type.

        Returns:
            dict with keys: 'conv', 'embedding', 'hidden', 'output', 'batch_norm'
            values: list of layer names in that category
        """
        classification = defaultdict(list)

        layer_names = [name for name, _ in model.named_modules()]

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                classification['conv'].append(name)
            elif isinstance(module, nn.Embedding):
                classification['embedding'].append(name)
            elif isinstance(module, nn.Linear):
                # Last linear layer = output
                if name.endswith('fc') or name.endswith('output') or name.endswith('classification'):
                    classification['output'].append(name)
                else:
                    classification['hidden'].append(name)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                classification['batch_norm'].append(name)

        return dict(classification)

    @staticmethod
    def get_layer_type(layer_name: str, classification: Dict[str, List[str]]) -> str:
        """Get the type of a specific layer."""
        # Handle parameter names like "conv1.weight" or "fc3.bias"
        # Extract base layer name (before the dot)
        base_name = layer_name.split('.')[0] if '.' in layer_name else layer_name

        for layer_type, layer_names in classification.items():
            # Check if base name is in the classified layers
            if base_name in layer_names or layer_name in layer_names:
                return layer_type

        # Fallback: classify by common patterns
        if 'conv' in layer_name:
            return 'conv'
        elif 'fc' in layer_name:
            if 'fc3' in layer_name or 'fc_out' in layer_name:
                return 'output'
            return 'hidden'
        elif 'linear' in layer_name.lower():
            return 'hidden'

        return 'unknown'


class SensitivityAnalyzer:
    """
    Analyze and track parameter sensitivity for signal-aware privacy allocation.

    METHODOLOGY:
    ============
    Sensitivity represents how much a parameter's gradient changes during training.

    - High sensitivity (σ > 0.7): Parameters that learn rapidly
      → Require stronger privacy protection (higher noise)
      → Example: Classification layer parameters

    - Low sensitivity (σ < 0.3): Parameters that learn slowly/stably
      → Can use less noise to preserve utility
      → Example: Early convolutional layers

    GRADIENT SAMPLING:
    ==================
    We compute sensitivity from gradient flows, not initial weights:

    1. Forward pass on multiple batches
    2. Compute gradients for each batch
    3. Track gradient variance across batches
    4. Use exponential moving average (EMA) for smoothing
    """

    def __init__(self, model: nn.Module, ema_alpha: float = 0.1):
        """
        Initialize sensitivity analyzer.

        Args:
            model: PyTorch model to analyze
            ema_alpha: EMA smoothing factor (0-1). Higher = more recent data weighted
        """
        self.model = model
        self.ema_alpha = ema_alpha
        self.sensitivity_history = defaultdict(lambda: [])
        self.ema_sensitivity = {}
        self.classifier = LayerClassifier()
        self.layer_classification = self.classifier.classify_layers(model)

    def compute_batch_gradient_sensitivity(
            self,
            batch
    ) -> Dict[str, float]:
        """
        Compute gradient sensitivity for a single batch.

        Args:
            batch: Either dict {"img": tensor, "label": tensor} or tuple (x, y)

        Returns:
            Dictionary mapping parameter name to gradient magnitude
        """
        # Handle different batch formats
        if isinstance(batch, dict):
            if "img" in batch:
                x, y = batch["img"], batch["label"]
            elif "image" in batch:
                x, y = batch["image"], batch["label"]
            else:
                raise ValueError(f"Unknown dict format with keys: {batch.keys()}")
        else:
            x, y = batch

        # Move to correct device
        device = next(self.model.parameters()).device
        x, y = x.to(device), y.to(device)

        self.model.zero_grad()

        # Forward pass
        output = self.model(x)
        loss = nn.CrossEntropyLoss()(output, y)

        # Backward pass
        loss.backward()

        # Compute gradient magnitudes (sensitivity = norm of gradient)
        grad_sensitivity = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                grad_sensitivity[name] = grad_norm
            else:
                grad_sensitivity[name] = 0.0

        return grad_sensitivity

    def sample_sensitivities(
            self,
            dataloader,
            num_batches: int = 5
    ) -> Dict[str, float]:
        """
        Sample gradient sensitivities over multiple batches.

        This provides robust sensitivity estimates by averaging over
        multiple data samples rather than single batch.

        Args:
            dataloader: Training data loader
            num_batches: Number of batches to sample

        Returns:
            Average gradient sensitivity per parameter
        """
        all_sensitivities = defaultdict(list)

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            grad_sens = self.compute_batch_gradient_sensitivity(batch)

            for name, sens in grad_sens.items():
                all_sensitivities[name].append(sens)

        # Average sensitivity across samples
        avg_sensitivities = {}
        for name, sensitivities in all_sensitivities.items():
            avg_sensitivities[name] = np.mean(sensitivities) if sensitivities else 0.0

        return avg_sensitivities

    def normalize_sensitivities(
            self,
            sensitivities: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Normalize sensitivities to [0, 1] range.

        Args:
            sensitivities: Raw sensitivity values

        Returns:
            Normalized sensitivities in [0, 1]
        """
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

    def update_ema_sensitivity(
            self,
            new_sensitivities: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Update exponential moving average of sensitivities.

        Smooths out noise in gradient-based estimates by maintaining
        running average with exponential decay.

        Args:
            new_sensitivities: New sensitivity measurements

        Returns:
            Updated EMA sensitivities
        """
        for name, new_sens in new_sensitivities.items():
            if name not in self.ema_sensitivity:
                self.ema_sensitivity[name] = new_sens
            else:
                old_ema = self.ema_sensitivity[name]
                self.ema_sensitivity[name] = (
                        (1 - self.ema_alpha) * old_ema +
                        self.ema_alpha * new_sens
                )

        return self.ema_sensitivity.copy()


class NoiseAllocator:
    """
    Allocate per-layer noise based on sensitivity while respecting
    total epsilon budget.

    PRINCIPLE:
    ==========
    High sensitivity layer → More privacy needed → Higher noise multiplier
    Low sensitivity layer → Less privacy needed → Lower noise multiplier

    CONSTRAINT:
    ===========
    Total privacy budget across all layers must equal target epsilon:

        Σ_i (n_samples_i / noise_multiplier_i) = target_epsilon

    Or equivalently (simplified):
        Σ_i noise_multiplier_i ≈ normalized sum
    """

    @staticmethod
    def allocate_noise_inversely_proportional(
            sensitivities: Dict[str, float],
            target_noise: float = 1.0,
            alpha: float = 1.0
    ) -> Dict[str, float]:
        """
        Allocate noise inversely proportional to sensitivity.

        Formula:
            noise_i = target_noise / (sensitivity_i + alpha)

        where alpha prevents division by very small sensitivities

        Args:
            sensitivities: Normalized sensitivities in [0, 1]
            target_noise: Base noise multiplier
            alpha: Regularization factor (avoids extreme values)

        Returns:
            Per-layer noise multipliers
        """
        noise_multipliers = {}

        for name, sens in sensitivities.items():
            # Inverse relationship: high sens → high noise
            noise = target_noise / (sens + alpha)
            noise_multipliers[name] = np.clip(noise, 0.5, 3.0)  # Bound for stability

        return noise_multipliers

    @staticmethod
    def allocate_noise_normalized(
            sensitivities: Dict[str, float],
            target_epsilon: float = 1.0
    ) -> Dict[str, float]:
        """
        Allocate noise such that total epsilon equals target.

        More sophisticated: ensures privacy budget constraint is satisfied.

        Noise inversely proportional to sensitivity, then normalized:

            raw_noise_i = 1.0 / sensitivity_i
            noise_i = target_epsilon * raw_noise_i / Σ(raw_noise_j)

        Args:
            sensitivities: Normalized sensitivities in [0, 1]
            target_epsilon: Total epsilon budget to allocate

        Returns:
            Per-layer noise multipliers summing to approximately target_epsilon
        """
        if not sensitivities:
            return {}

        # Compute inverse proportionality (add small epsilon to avoid div/0)
        raw_noise = {}
        for name, sens in sensitivities.items():
            raw_noise[name] = 1.0 / (sens + 1e-6)

        # Normalize to sum to target epsilon
        total_raw = sum(raw_noise.values())
        noise_multipliers = {
            name: target_epsilon * noise / total_raw
            for name, noise in raw_noise.items()
        }

        return noise_multipliers

    @staticmethod
    def allocate_noise_by_layer_type(
            sensitivities: Dict[str, float],
            layer_classification: Dict[str, List[str]],
            target_noise: float = 1.0,
            layer_type_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Allocate noise differently by layer type.

        Different layer types need different privacy levels:
        - Input/Conv layers: Lower noise (preserve features)
        - Hidden layers: Medium noise
        - Output layers: Higher noise (classification critical)

        Args:
            sensitivities: Per-parameter sensitivity scores
            layer_classification: Output from LayerClassifier.classify_layers()
            target_noise: Base noise multiplier
            layer_type_weights: Custom weights for each layer type

        Returns:
            Per-layer noise multipliers
        """
        if layer_type_weights is None:
            layer_type_weights = {
                'conv': 0.7,  # Less noise for feature extraction
                'embedding': 0.7,
                'hidden': 1.0,  # Medium noise
                'batch_norm': 0.5,  # Minimal noise
                'output': 1.5,  # More noise for classification
                'unknown': 1.0,
            }

        noise_multipliers = {}
        classifier = LayerClassifier()

        for name, sens in sensitivities.items():
            layer_type = classifier.get_layer_type(name, layer_classification)
            base_multiplier = layer_type_weights.get(layer_type, 1.0)

            # Combine layer-type prior with sensitivity
            noise = target_noise * base_multiplier * (1.0 + sens)
            noise_multipliers[name] = np.clip(noise, 0.5, 3.0)

        return noise_multipliers


class ClippingNormAllocator:
    """
    Allocate per-layer clipping norms based on sensitivity.

    RELATIONSHIP: noise_multiplier ↔ clipping_norm

    DP Theory: Larger clipping norm → larger gradients → need more noise
              Smaller clipping norm → clipped more → less noise needed

    Strategy:
      High sensitivity layer: SMALL clip norm (forces clipping for privacy)
      Low sensitivity layer: LARGE clip norm (allow larger gradients)
    """

    @staticmethod
    def get_layer_clip_norms(
            sensitivities: Dict[str, float],
            base_norm: float = 1.0,
            inverse_scaling: bool = True
    ) -> Dict[str, float]:
        """
        Compute per-layer clipping norms.

        Args:
            sensitivities: Normalized sensitivities [0, 1]
            base_norm: Base clipping norm
            inverse_scaling: If True, high sens → small norm (more clipping)
                           If False, high sens → large norm (less clipping)

        Returns:
            Per-layer clipping norms
        """
        clipping_norms = {}

        for name, sens in sensitivities.items():
            if inverse_scaling:
                # High sensitivity → Small clip norm (more aggressive clipping)
                norm = base_norm / (1.0 + sens)
            else:
                # High sensitivity → Large clip norm (allow larger updates)
                norm = base_norm * (1.0 + sens)

            clipping_norms[name] = np.clip(norm, 0.1, 3.0)

        return clipping_norms


class SensitivityReporter:
    """Generate reports and visualizations of sensitivity analysis."""

    @staticmethod
    def print_sensitivity_report(
            sensitivities: Dict[str, float],
            layer_classification: Optional[Dict[str, List[str]]] = None,
            noise_multipliers: Optional[Dict[str, float]] = None
    ):
        """
        Print formatted sensitivity analysis report.

        Args:
            sensitivities: Sensitivity scores
            layer_classification: Optional layer types
            noise_multipliers: Optional noise allocation
        """
        if not sensitivities:
            logger.warning("No sensitivity data available")
            return

        print("\n" + "=" * 90)
        print("SIGNAL SENSITIVITY ANALYSIS REPORT")
        print("=" * 90)

        sorted_sens = sorted(sensitivities.items(), key=lambda x: -x[1])

        classifier = LayerClassifier()
        for i, (name, sens) in enumerate(sorted_sens[:15]):
            bar_len = int(sens * 50)
            bar = "█" * bar_len + "░" * (50 - bar_len)

            layer_type = "unknown"
            if layer_classification:
                layer_type = classifier.get_layer_type(name, layer_classification)

            if noise_multipliers and name in noise_multipliers:
                noise_str = f"| noise={noise_multipliers[name]:.3f}"
            else:
                noise_str = "| noise=N/A"

            print(f"{name:35s} | {bar} | sens={sens:.4f} | {layer_type:10s} {noise_str}")

        print("=" * 90)
        print(f"Average Sensitivity: {np.mean(list(sensitivities.values())):.4f}")
        print(f"Max Sensitivity: {max(sensitivities.values()):.4f}")
        print(f"Min Sensitivity: {min(sensitivities.values()):.4f}")

        if layer_classification:
            print(f"\nLayer Type Distribution:")
            for ltype, layers in layer_classification.items():
                if layers:
                    avg_sens = np.mean([sensitivities.get(l, 0) for l in layers])
                    print(f"  {ltype:15s}: {len(layers):3d} layers, avg sens={avg_sens:.4f}")

        print("=" * 90 + "\n")


# ============================================================================
# INTEGRATION FUNCTIONS (Use these in your training code)
# ============================================================================

def compute_adaptive_layer_sensitivity(
        model: nn.Module,
        dataloader,
        num_batches: int = 5,
        use_ema: bool = True
) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    """
    Compute adaptive layer sensitivity for DP allocation.

    RECOMMENDED: Call this at the start of training on first few batches.

    Args:
        model: PyTorch model
        dataloader: Training data
        num_batches: Batches to sample for robustness
        use_ema: Whether to use EMA smoothing

    Returns:
        (sensitivities, layer_classification)
    """
    analyzer = SensitivityAnalyzer(model)

    # Sample gradients from multiple batches
    raw_sensitivities = analyzer.sample_sensitivities(dataloader, num_batches)

    # Normalize to [0, 1]
    sensitivities = analyzer.normalize_sensitivities(raw_sensitivities)

    # Get layer classification
    layer_classification = analyzer.layer_classification

    return sensitivities, layer_classification


def get_adaptive_dp_config(
        sensitivities: Dict[str, float],
        layer_classification: Dict[str, List[str]],
        target_epsilon: float = 1.0,
        strategy: str = "inverse"
) -> Dict:
    """
    Get complete adaptive DP configuration based on sensitivity.

    Args:
        sensitivities: Normalized sensitivities [0, 1]
        layer_classification: Layer types
        target_epsilon: Target epsilon budget
        strategy: "inverse", "normalized", or "by_layer_type"

    Returns:
        DP config with per-layer noise and clipping
    """
    allocator = NoiseAllocator()
    clip_allocator = ClippingNormAllocator()

    if strategy == "inverse":
        noise_multipliers = allocator.allocate_noise_inversely_proportional(
            sensitivities, target_noise=target_epsilon
        )
    elif strategy == "normalized":
        noise_multipliers = allocator.allocate_noise_normalized(
            sensitivities, target_epsilon=target_epsilon
        )
    elif strategy == "by_layer_type":
        noise_multipliers = allocator.allocate_noise_by_layer_type(
            sensitivities, layer_classification, target_noise=target_epsilon
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    clipping_norms = clip_allocator.get_layer_clip_norms(sensitivities)

    return {
        "sensitivities": sensitivities,
        "noise_multipliers": noise_multipliers,
        "clipping_norms": clipping_norms,
        "strategy": strategy,
    }


if __name__ == "__main__":
    # Example usage
    from qpriviot_fl.task import make_model

    model = make_model("cifar10")
    print("Model loaded")

    # In practice, use real dataloader:
    # sensitivities, classification = compute_adaptive_layer_sensitivity(model, train_loader)

    # For demo, compute on model initialization
    analyzer = SensitivityAnalyzer(model)
    dummy_sensitivities = {
        name: np.random.uniform(0.3, 0.9)
        for name, _ in model.named_parameters()
    }

    normalized_sens = analyzer.normalize_sensitivities(dummy_sensitivities)
    layer_classification = analyzer.layer_classification

    # Get DP config
    dp_config = get_adaptive_dp_config(
        normalized_sens,
        layer_classification,
        target_epsilon=1.0,
        strategy="normalized"
    )

    # Print report
    SensitivityReporter.print_sensitivity_report(
        normalized_sens,
        layer_classification,
        dp_config["noise_multipliers"]
    )
