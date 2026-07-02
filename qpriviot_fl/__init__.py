"""QPrivIoT-FL: shared privacy/crypto substrate for the agent-memory experiments.

The Flower FL app (client_app / server_app / task / device_profile / config) that
originally lived here was archived to scripts/archive/flower_app/ when the project
pivoted from the (scooped) QFL direction to federated agent-memory. Only privacy_utils
— the SecAgg + Skellam crypto reused by the active experiments and tests — remains.
To run the old `flwr run` FL app again, restore those modules into this package.
"""

__version__ = "0.1.0"

from qpriviot_fl.privacy_utils import (
    RenyiPrivacyAccountant,
    allocate_adaptive_noise,
    apply_dp_noise_per_layer,
)

__all__ = [
    "RenyiPrivacyAccountant",
    "allocate_adaptive_noise",
    "apply_dp_noise_per_layer",
]
