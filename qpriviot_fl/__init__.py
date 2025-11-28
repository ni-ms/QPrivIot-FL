"""QPrivIoT-FL: Adaptive Privacy-Preserving Federated Learning for IoT Devices"""

__version__ = "0.1.0"


from qpriviot_fl.config import DEFAULT_CONFIG
from qpriviot_fl.device_profile import profile_device
from qpriviot_fl.task import make_model, load_data, train, test
from qpriviot_fl.privacy_utils import (
    RenyiPrivacyAccountant,
    compute_gradient_sensitivity,
    allocate_adaptive_noise,
    apply_dp_noise_per_layer,
)

__all__ = [
    "DEFAULT_CONFIG",
    "profile_device",
    "make_model",
    "load_data",
    "train",
    "test",
    "RenyiPrivacyAccountant",
    "compute_gradient_sensitivity",
    "allocate_adaptive_noise",
    "apply_dp_noise_per_layer",
]
