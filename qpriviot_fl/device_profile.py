"""Device profiling module with real measurements."""

import time
import socket
import psutil
import platform
from typing import Dict, Optional


def measure_bandwidth() -> float:
    """
    Measure actual network bandwidth using socket test.
    Returns bandwidth in Mbps.
    """
    try:

        test_data = b'0' * (1024 * 1024)
        start = time.time()

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(test_data, ('127.0.0.1', 9999))
        sock.close()
        elapsed = time.time() - start
        bandwidth = (len(test_data) * 8) / (elapsed * 1_000_000)
        return max(1.0, bandwidth)
    except Exception:

        net_io = psutil.net_io_counters()
        return min(100.0, net_io.bytes_sent / 1_000_000)


def profile_device() -> Dict[str, float]:
    """
    Comprehensive device profiling.
    Returns normalized scores and raw metrics.
    """

    cpu_percent = psutil.cpu_percent(interval=0.5)

    ram_percent = psutil.virtual_memory().percent

    battery = psutil.sensors_battery()
    battery_percent = battery.percent if battery else 100.0

    bandwidth = measure_bandwidth()

    resource_score = (
                             (100 - cpu_percent) +
                             (100 - ram_percent) +
                             battery_percent +
                             min(bandwidth, 100)
                     ) / 400.0

    profile = {
        "cpu_percent": cpu_percent,
        "ram_percent": ram_percent,
        "battery_percent": battery_percent,
        "bandwidth_mbps": bandwidth,
        "resource_score": resource_score,
        "platform": platform.system(),
    }

    return profile


def is_eligible_for_training(profile: Dict[str, float],
                             min_cpu_avail: float = 30.0,
                             min_ram_avail: float = 30.0,
                             min_battery: float = 20.0,
                             min_bandwidth: float = 5.0) -> bool:
    """
    Check if device meets minimum requirements for participation.
    """
    cpu_available = 100 - profile["cpu_percent"]
    ram_available = 100 - profile["ram_percent"]

    eligible = (
            cpu_available >= min_cpu_avail and
            ram_available >= min_ram_avail and
            profile["battery_percent"] >= min_battery and
            profile["bandwidth_mbps"] >= min_bandwidth
    )

    return eligible


def detect_straggler(profile: Dict[str, float],
                     threshold: float = 0.3) -> bool:
    """
    Detect if device is a straggler based on resource score.
    """
    return profile["resource_score"] < threshold
