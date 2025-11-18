"""Self-contained device profiling with realistic IoT simulation."""

import psutil
import platform
import random
import os
import threading
from typing import Dict

IOT_DEVICES = {
    "raspberry_pi_4": {
        "cpu": (15, 35),
        "ram": (45, 65),
        "battery": (90, 100),
        "bandwidth": (25, 50),
        "probability": 0.25,
    },
    "raspberry_pi_zero": {
        "cpu": (40, 70),
        "ram": (60, 80),
        "battery": (40, 80),
        "bandwidth": (8, 20),
        "probability": 0.20,
    },
    "smartphone": {
        "cpu": (20, 50),
        "ram": (50, 75),
        "battery": (30, 85),
        "bandwidth": (15, 80),
        "probability": 0.25,
    },
    "edge_server": {
        "cpu": (10, 25),
        "ram": (30, 50),
        "battery": (100, 100),
        "bandwidth": (60, 150),
        "probability": 0.15,
    },
    "iot_sensor": {
        "cpu": (50, 85),
        "ram": (65, 90),
        "battery": (20, 60),
        "bandwidth": (3, 12),
        "probability": 0.15,
    },
}

_thread_local = threading.local()


def _get_client_id() -> int:
    """
    Auto-detect client ID from process/thread.
    Each client process gets a unique, consistent ID.
    """
    if not hasattr(_thread_local, 'client_id'):
        pid = os.getpid()
        tid = threading.get_ident()
        _thread_local.client_id = (pid + tid) % 1000

    return _thread_local.client_id


def profile_device() -> Dict[str, float]:
    """
    Self-contained device profiling.
    Automatically creates realistic mock IoT device based on caller.
    NO PARAMETERS NEEDED!
    """

    client_id = _get_client_id()

    random.seed(f"device_type_{client_id}")
    types = list(IOT_DEVICES.keys())
    probs = [IOT_DEVICES[t]["probability"] for t in types]
    device_type = random.choices(types, weights=probs, k=1)[0]

    random.seed(f"metrics_{client_id}")
    device = IOT_DEVICES[device_type]

    cpu_percent = random.uniform(*device["cpu"])
    ram_percent = random.uniform(*device["ram"])
    battery_percent = random.uniform(*device["battery"])
    bandwidth_mbps = random.uniform(*device["bandwidth"])

    random.seed()

    resource_score = (
                             (100 - cpu_percent) * 0.25 +
                             (100 - ram_percent) * 0.25 +
                             battery_percent * 0.25 +
                             min(bandwidth_mbps, 100) * 0.25
                     ) / 100.0

    return {
        "cpu_percent": cpu_percent,
        "ram_percent": ram_percent,
        "battery_percent": battery_percent,
        "bandwidth_mbps": bandwidth_mbps,
        "resource_score": resource_score,
        "device_type": device_type,
        "platform": platform.system(),
    }


def is_eligible_for_training(
        profile: Dict[str, float],
        min_cpu_avail: float = 5.0,
        min_ram_avail: float = 5.0,
        min_battery: float = 15.0,
        min_bandwidth: float = 2.0
) -> bool:
    """
    Check if device can participate.
    VERY PERMISSIVE - almost all devices pass.
    """
    cpu_available = 100 - profile["cpu_percent"]
    ram_available = 100 - profile["ram_percent"]

    eligible = (
            cpu_available >= min_cpu_avail and
            ram_available >= min_ram_avail and
            profile["battery_percent"] >= min_battery and
            profile["bandwidth_mbps"] >= min_bandwidth
    )

    if not eligible:
        print(f" Device {profile.get('device_type', 'unknown')} dropped: "
              f"cpu_avail={cpu_available:.1f}%, ram_avail={ram_available:.1f}%, "
              f"battery={profile['battery_percent']:.1f}%, bw={profile['bandwidth_mbps']:.1f}Mbps")

    return eligible


def detect_straggler(profile: Dict[str, float], threshold: float = 0.3) -> bool:
    """
    Detect if device is a straggler (weak resources).
    """
    is_straggler = profile["resource_score"] < threshold

    if is_straggler:
        print(f" Straggler: {profile.get('device_type', 'unknown')} (score={profile['resource_score']:.2f})")

    return is_straggler


def get_device_aware_noise_multiplier(profile: Dict[str, float]) -> float:
    """
    Allocate noise multiplier based on device type and resources.

    Phase 2: Device-Aware Privacy Allocation
    - Low-power devices (RPi Zero, IoT sensors) → less noise (better utility)
    - Medium-power devices (Smartphone, RPi 4) → balanced noise
    - High-power devices (Edge server) → more noise (stronger privacy)

    Args:
        profile: Device profile from profile_device()

    Returns:
        noise_multiplier: Float value for DP noise (0.8 to 1.2)
    """
    device_type = profile.get("device_type", "unknown")
    resource_score = profile.get("resource_score", 0.5)

    # Device-based allocation (Phase 2)
    if device_type == "raspberry_pi_zero" or device_type == "iot_sensor":
        noise_multiplier = 0.8
        print(f" Low-power device ({device_type}): noise_mult={noise_multiplier:.2f}")

    elif device_type == "smartphone" or device_type == "raspberry_pi_4":
        noise_multiplier = 1.0
        print(f" Medium-power device ({device_type}): noise_mult={noise_multiplier:.2f}")

    elif device_type == "edge_server":
        noise_multiplier = 1.2
        print(f" High-power device ({device_type}): noise_mult={noise_multiplier:.2f}")

    else:
        if resource_score < 0.4:
            noise_multiplier = 0.8
        elif resource_score < 0.7:
            noise_multiplier = 1.0
        else:
            noise_multiplier = 1.2
        print(f" Unknown device: resource_score={resource_score:.2f} → noise={noise_multiplier:.2f}")

    return noise_multiplier
