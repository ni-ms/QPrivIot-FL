"""QPrivIot-FL: Enhanced client with energy tracking and adaptive features."""
import torch
from flwr.client import ClientApp
from flwr.common import ArrayRecord, MetricRecord, RecordDict, Message, Context
import psutil
import numpy as np
import time

from qpriviot_fl.task import (
    Net, FEMNISTNet, load_data,
    train_with_adaptive_dp, test,
    signal_sensitivity_estimation
)

app = ClientApp()


class EnergyTracker:
    """Track energy consumption during training."""

    def __init__(self):
        self.start_time = None
        self.start_battery = None
        self.energy_consumed = 0.0

    def start(self):
        """Start tracking energy."""
        self.start_time = time.time()
        battery = psutil.sensors_battery()
        self.start_battery = battery.percent if battery else 100.0

    def stop(self):
        """Stop tracking and calculate energy consumed."""
        end_time = time.time()
        battery = psutil.sensors_battery()
        end_battery = battery.percent if battery else self.start_battery

        battery_drain = self.start_battery - end_battery
        training_time = end_time - self.start_time

        self.energy_consumed = max(0.0, battery_drain)

        return {
            "energy_consumed": self.energy_consumed,
            "training_time_sec": training_time,
            "battery_drain_percent": battery_drain
        }


def device_profiling():
    """Enhanced device profiling: CPU, RAM, battery, network bandwidth."""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem_percent = psutil.virtual_memory().percent
        battery = psutil.sensors_battery()
        battery_percent = battery.percent if battery else 100.0

        cpu_count = psutil.cpu_count()

        available_mem_gb = psutil.virtual_memory().available / (1024 ** 3)

    except Exception:
        cpu_percent = 50.0
        mem_percent = 50.0
        battery_percent = 100.0
        cpu_count = 4
        available_mem_gb = 4.0

    net_speed = np.random.uniform(1.0, 10.0)

    resource_score = (
            (100 - cpu_percent) / 100 * 0.3 +
            (100 - mem_percent) / 100 * 0.2 +
            battery_percent / 100 * 0.3 +
            min(net_speed / 10.0, 1.0) * 0.2
    )

    return {
        "cpu_percent": cpu_percent,
        "mem_percent": mem_percent,
        "battery_percent": battery_percent,
        "net_speed_mbps": net_speed,
        "cpu_count": cpu_count,
        "available_mem_gb": available_mem_gb,
        "resource_score": resource_score,
    }


def adjust_local_epochs(dp_profile, base_epochs=1):
    """Dynamically adjust local epochs based on device resources."""
    battery = dp_profile["battery_percent"]
    cpu_usage = dp_profile["cpu_percent"]

    if battery < 20:
        return 1
    elif battery < 50:
        return max(1, base_epochs - 1)
    elif cpu_usage > 80:
        return max(1, base_epochs - 1)
    else:
        return base_epochs


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data with adaptive DP and energy tracking."""

    dataset_name = context.run_config.get("dataset", "cifar10")

    if dataset_name == "cifar10":
        model = Net()
    elif dataset_name == "femnist":
        model = FEMNISTNet()
    else:
        model = Net()

    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _, img_key, label_key = load_data(partition_id, num_partitions, dataset_name)

    dp_profile = device_profiling()

    energy_tracker = EnergyTracker()
    energy_tracker.start()

    global_sensitivity, layer_sensitivity = signal_sensitivity_estimation(
        trainloader, model, device, img_key, label_key
    )

    privacy_budget = msg.content["config"].get("privacy_budget", 2.0)
    lr = msg.content["config"].get("lr", 0.001)
    base_local_epochs = context.run_config.get("local-epochs", 1)

    local_epochs = adjust_local_epochs(dp_profile, base_local_epochs)

    base_noise = 1.0 / privacy_budget

    resource_factor = 0.8 + (1.0 - dp_profile["resource_score"]) * 0.4

    noise_multiplier = base_noise * global_sensitivity * resource_factor

    noise_multiplier = np.clip(noise_multiplier, 0.1, 5.0)

    max_grad_norm = 1.0
    target_delta = 1e-5

    train_loss, train_acc, epsilon, best_alpha, _ = train_with_adaptive_dp(
        model,
        trainloader,
        epochs=local_epochs,
        lr=lr,
        device=device,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        target_delta=target_delta,
        layer_sensitivity=layer_sensitivity,
        img_key=img_key,
        label_key=label_key,
    )

    energy_metrics = energy_tracker.stop()

    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "num-examples": len(trainloader.dataset),
        "dp_epsilon": epsilon,
        "dp_alpha": best_alpha,
        "noise_multiplier": noise_multiplier,
        "global_sensitivity": global_sensitivity,
        "local_epochs_used": local_epochs,
        **dp_profile,
        **energy_metrics,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    dataset_name = context.run_config.get("dataset", "cifar10")

    if dataset_name == "cifar10":
        model = Net()
    elif dataset_name == "femnist":
        model = FEMNISTNet()
    else:
        model = Net()

    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader, img_key, label_key = load_data(partition_id, num_partitions, dataset_name)

    eval_loss, eval_acc = test(model, valloader, device, img_key, label_key)

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})

    return Message(content=content, reply_to=msg)
