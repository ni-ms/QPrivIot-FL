import random
import threading
import os
from qpriviot_fl.config import DEFAULT_CONFIG

_thread_local = threading.local()


def profile_device(config=DEFAULT_CONFIG):
    """
    Profiles device capability (0.0 - 1.0) and status.
    Returns device profile vector and readiness score.
    References: AdaPriv Section 3.2
    """
    if not hasattr(_thread_local, 'client_id'):
        _thread_local.client_id = (os.getpid() + threading.get_ident()) % 1000

    random.seed(f"device_profile_{_thread_local.client_id}")

    device_types = list(config.resource.device_distribution.keys())
    device_probs = list(config.resource.device_distribution.values())
    device_type = random.choices(device_types, weights=device_probs)[0]

    # cap_i, mem_i, batt_i, net_i as per Section 3.2
    base_score = config.resource.device_scores[device_type]
    
    cap_i = max(0.05, min(1.0, base_score + random.uniform(-0.05, 0.05)))
    mem_i = max(0.1, min(1.0, base_score + random.uniform(-0.1, 0.1)))
    batt_i = random.uniform(0.1, 1.0)  # Simulated battery status
    net_i = random.uniform(0.2, 1.0)   # Simulated network status

    # Readiness Score R_i^t = min(cap_i, mem_i, batt_i * alpha_batt, net_i * alpha_net)
    alpha_batt = 0.7
    alpha_net = 0.8
    readiness_score = min(cap_i, mem_i, batt_i * alpha_batt, net_i * alpha_net)

    return {
        "device_type": device_type,
        "resource_score": readiness_score,
        "profile": {
            "cap": cap_i,
            "mem": mem_i,
            "batt": batt_i,
            "net": net_i
        }
    }
