"""Flower client with SecAgg, adaptive DP, and device profiling - FINAL FIX."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context

from qpriviot_fl.task import make_model, get_weights, set_weights, load_data, train, test
from qpriviot_fl.device_profile import profile_device, is_eligible_for_training
from qpriviot_fl.sensitivity import analyze_model_sensitivity, get_average_sensitivity
from qpriviot_fl.privacy import compute_adaptive_dp_config


class QPrivIoTClient(NumPyClient):
    """FL client with adaptive privacy and resource awareness."""

    def __init__(
            self,
            trainloader,
            valloader,
            local_epochs: int,
            learning_rate: float,
            dataset_name: str,
            total_rounds: int,
    ):
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.dataset_name = dataset_name
        self.total_rounds = total_rounds
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.current_round = 0

    def fit(self, parameters, config):
        """Train with adaptive DP based on device profile and sensitivity."""

        device_profile = profile_device()

        if not is_eligible_for_training(device_profile):
            print(f"Client dropped: resource_score={device_profile['resource_score']:.2f}")

            dummy_model = make_model(self.dataset_name)
            return get_weights(dummy_model), len(self.trainloader.dataset), {
                "dropped": 1,
                "resource_score": device_profile['resource_score'],
                "cpu_percent": device_profile['cpu_percent'],
                "ram_percent": device_profile['ram_percent'],
                "battery_percent": device_profile['battery_percent'],
                "bandwidth_mbps": device_profile['bandwidth_mbps'],
            }

        model = make_model(self.dataset_name)
        set_weights(model, parameters)

        sensitivities = analyze_model_sensitivity(model)
        avg_sensitivity = get_average_sensitivity(sensitivities)

        convergence_score = config.get("convergence_score", 0.0)
        self.current_round = config.get("current_round", 0)

        dp_config = compute_adaptive_dp_config(
            device_profile=device_profile,
            sensitivities=sensitivities,
            convergence_score=convergence_score,
            current_round=self.current_round,
            total_rounds=self.total_rounds,
        )

        results = train(
            model,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            self.device,
            dp_config=dp_config,
            dataset_name=self.dataset_name,
        )

        metrics = {
            "train_loss": float(results.get("train_loss", 0.0)),
            "val_loss": float(results.get("val_loss", 0.0)),
            "val_accuracy": float(results.get("val_accuracy", 0.0)),
            "epsilon": float(results.get("epsilon", 0.0)) if results.get("epsilon") is not None else 0.0,
            "resource_score": float(device_profile["resource_score"]),
            "cpu_percent": float(device_profile["cpu_percent"]),
            "ram_percent": float(device_profile["ram_percent"]),
            "battery_percent": float(device_profile["battery_percent"]),
            "bandwidth_mbps": float(device_profile["bandwidth_mbps"]),
            "avg_sensitivity": float(avg_sensitivity),
            "noise_multiplier": float(dp_config["noise_multiplier"]),
            "max_grad_norm": float(dp_config["max_grad_norm"]),
        }

        metrics = {k: v for k, v in metrics.items() if v is not None}

        return get_weights(model), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        """Evaluate model."""
        model = make_model(self.dataset_name)
        set_weights(model, parameters)
        loss, accuracy = test(model, self.valloader, self.device, self.dataset_name)

        return float(loss), len(self.valloader.dataset), {
            "accuracy": float(accuracy)
        }


def client_fn(context: Context):
    """Create client instance."""

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config.get("batch-size", 32)
    dataset_name = context.run_config.get("dataset", "cifar10")

    trainloader, valloader = load_data(
        partition_id,
        num_partitions,
        batch_size,
        dataset_name
    )

    local_epochs = context.run_config.get("local-epochs", 2)
    lr = context.run_config.get("learning-rate", 0.001)
    total_rounds = context.run_config.get("num-server-rounds", 10)

    return QPrivIoTClient(
        trainloader,
        valloader,
        local_epochs,
        lr,
        dataset_name,
        total_rounds,
    ).to_client()


app = ClientApp(
    client_fn=client_fn,
    mods=[secaggplus_mod],
)
