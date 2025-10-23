"""Flower client with SecAgg, adaptive DP, and device profiling."""

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
        self.model = make_model(dataset_name)
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
            print(f"Client dropped due to insufficient resources: {device_profile['resource_score']:.2f}")

            return get_weights(self.model), len(self.trainloader.dataset), {
                "dropped": True,
                **device_profile
            }

        set_weights(self.model, parameters)

        sensitivities = analyze_model_sensitivity(self.model)
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
            self.model,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            self.device,
            dp_config=dp_config,
            dataset_name=self.dataset_name,
        )

        metrics = {
            **results,
            **device_profile,
            "avg_sensitivity": avg_sensitivity,
            "noise_multiplier": dp_config["noise_multiplier"],
            "max_grad_norm": dp_config["max_grad_norm"],
        }

        return get_weights(self.model), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        """Evaluate model."""
        set_weights(self.model, parameters)
        loss, accuracy = test(self.model, self.valloader, self.device, self.dataset_name)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


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
