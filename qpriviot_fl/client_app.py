import torch
from flwr.client import ClientApp

from flwr.common import ArrayRecord, MetricRecord, RecordDict, Message, Context
import psutil
import numpy as np


from qpriviot_fl.task import Net, load_data, train_with_dp, test, signal_sensitivity_estimation

app = ClientApp()


def device_profiling():
    """Lightweight device profiling: CPU, RAM, battery, network bandwidth."""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem_percent = psutil.virtual_memory().percent
        battery = psutil.sensors_battery()
        battery_percent = battery.percent if battery else 100.0
    except Exception:
        cpu_percent = 50.0
        mem_percent = 50.0
        battery_percent = 100.0

    # Simulate network bandwidth
    net_speed = np.random.uniform(1.0, 10.0)

    return {
        "cpu_percent": cpu_percent,
        "mem_percent": mem_percent,
        "battery_percent": battery_percent,
        "net_speed_mbps": net_speed,
    }


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data with adaptive DP."""

    # Load model and initialize with received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load local data partition
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _ = load_data(partition_id, num_partitions)

    # Device profiling
    dp_profile = device_profiling()

    # Estimate signal sensitivity
    signal_sens = signal_sensitivity_estimation(trainloader)

    # Get privacy budget from server config
    privacy_budget = msg.content["config"].get("privacy_budget", 2.0)
    lr = msg.content["config"].get("lr", 0.001)

    # Adaptive noise multiplier based on device resources and sensitivity
    base_noise = 1.0 / privacy_budget
    resource_factor = 1.0 + (dp_profile["cpu_percent"] / 100.0) * 0.2
    noise_multiplier = base_noise * signal_sens * resource_factor

    # DP training parameters
    max_grad_norm = 1.0
    target_delta = 1e-5
    local_epochs = context.run_config.get("local-epochs", 1)

    # Train with DP-SGD
    train_loss, epsilon, best_alpha, _ = train_with_dp(
        model,
        trainloader,
        epochs=local_epochs,
        lr=lr,
        device=device,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        target_delta=target_delta,
    )

    # Prepare response message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "dp_epsilon": epsilon,
        "dp_alpha": best_alpha,
        "noise_multiplier": noise_multiplier,
        **dp_profile,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load model with received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load validation data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader = load_data(partition_id, num_partitions)

    # Evaluate
    eval_loss, eval_acc = test(model, valloader, device)

    # Prepare response
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})

    return Message(content=content, reply_to=msg)
