import random
import threading
import os
from qpriviot_fl.config import DEFAULT_CONFIG

_thread_local = threading.local()


def profile_device(config=DEFAULT_CONFIG):
    """
    Profiles device capability (0.0 - 1.0).
    Returns device type and resource score.
    """
    if not hasattr(_thread_local, 'client_id'):
        _thread_local.client_id = (os.getpid() + threading.get_ident()) % 1000

    random.seed(f"device_profile_{_thread_local.client_id}")

    device_types = list(config.resource.device_distribution.keys())
    device_probs = list(config.resource.device_distribution.values())
    device_type = random.choices(device_types, weights=device_probs)[0]

    base_score = config.resource.device_scores[device_type]
    fluctuation = random.uniform(-0.05, 0.05)
    resource_score = max(0.05, min(1.0, base_score + fluctuation))

    return {
        "device_type": device_type,
        "resource_score": resource_score
    }
