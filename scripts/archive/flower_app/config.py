from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PrivacyConfig:
    target_epsilon: float = 10.0
    target_delta: float = 1e-5
    initial_clip_norm: float = 1.0
    base_noise_multiplier: float = 1.0
    min_noise_multiplier: float = 0.5
    max_noise_multiplier: float = 5.0


@dataclass
class ResourceConfig:
    min_eligible_score: float = 0.05
    base_latency_seconds: float = 0.05

    secagg_overhead_factor: float = 2.5

    device_distribution: Dict[str, float] = field(default_factory=lambda: {
        "raspberry_pi_4": 0.3,
        "raspberry_pi_zero": 0.2,
        "smartphone": 0.3,
        "iot_sensor": 0.2
    })
    device_scores: Dict[str, float] = field(default_factory=lambda: {
        "raspberry_pi_4": 0.6,
        "raspberry_pi_zero": 0.2,
        "smartphone": 0.9,
        "iot_sensor": 0.1
    })


@dataclass
class AdaptivePrivacyConfig:
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    resource: ResourceConfig = field(default_factory=ResourceConfig)


DEFAULT_CONFIG = AdaptivePrivacyConfig()