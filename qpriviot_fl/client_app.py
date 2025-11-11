"""Flower client with SecAgg, adaptive DP, and device profiling - REVIEWED & FIXED."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context
from qpriviot_fl.device_profile import profile_device, is_eligible_for_training
from qpriviot_fl.sensitivity import SensitivityReporter, get_adaptive_dp_config, compute_adaptive_layer_sensitivity
from qpriviot_fl.task import make_model, get_weights, set_weights, load_data, train, test


class QPrivIoTClient(NumPyClient):
    """
    FL client with adaptive privacy and resource awareness.
    
    Features:
    - Device profiling for heterogeneous IoT environments
    - Sensitivity-based noise allocation
    - Convergence-aware privacy scheduling
    - Secure aggregation with SecAgg+
    """

    def __init__(
            self,
            trainloader,
            valloader,
            partition_id: int,
            local_epochs: int,
            learning_rate: float,
            dataset_name: str,
            total_rounds: int,
            device: str = "auto",
    ):
        self.trainloader = trainloader
        self.valloader = valloader
        self.partition_id = partition_id
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.dataset_name = dataset_name
        self.total_rounds = total_rounds
        self.current_round = 0
        self.last_valid_round = -1

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def fit(self, parameters, config):
        """
        Train model with configurable DP modes.

        Supports:
        - No DP (baseline)
        - Standard DP (fixed noise)
        - Adaptive DP (resource + sensitivity + convergence aware)

        Returns:
            Tuple of (weights, num_examples, metrics)
        """

        device_profile = profile_device()

        # Round counter with fallback
        config_round = int(config.get("current_round", -1))
        if config_round >= 0:
            self.current_round = config_round
            self.last_valid_round = config_round
        elif self.last_valid_round >= 0:
            self.current_round = self.last_valid_round + 1
            print(f"⚠️ Client {self.partition_id}: Using round fallback {self.current_round}")
        else:
            print(f"❌ Client {self.partition_id}: No valid round received")

        convergence_score = float(config.get("convergence_score", 0.0))

        use_dp = bool(config.get("use_dp", True))
        use_adaptive_dp = bool(config.get("use_adaptive_dp", True))

        if not is_eligible_for_training(device_profile):
            print(f"⚠️ Client {self.partition_id} [{device_profile.get('device_type', 'unknown')}] dropped: "
                  f"resource_score={device_profile['resource_score']:.2f}")

            model = make_model(self.dataset_name)
            model.to(self.device)
            set_weights(model, parameters)

            return get_weights(model), -1, {
                "train_loss": 0.0,
                "val_loss": 0.0,
                "val_accuracy": 0.0,
                "epsilon": 0.0,
                "device_type": device_profile.get("device_type", "unknown"),  # Only type
                "avg_sensitivity": 0.0,
                "noise_multiplier": 0.0,
                "max_grad_norm": 0.0,
                "dropped": 1,
            }

        # Create model ONCE (before any checks)
        model = make_model(self.dataset_name)
        model.to(self.device)
        set_weights(model, parameters)

        # Compute per-layer sensitivity BEFORE DP attachment
        sensitivities, layer_classification = compute_adaptive_layer_sensitivity(
            model,
            self.trainloader,
            num_batches=3,
            use_ema=True
        )

        avg_sensitivity = sum(sensitivities.values()) / max(len(sensitivities), 1)

        if not use_dp:
            dp_config = None
            dp_mode = "no_dp"
            print(f"🔓 Client {self.partition_id}: Training WITHOUT DP")

        elif not use_adaptive_dp:
            # Standard DP: global noise
            dp_config = {
                "noise_multiplier": 1.0,
                "max_grad_norm": 1.0,
                "per_layer": False,  # Flag for global DP
            }
            dp_mode = "standard_dp"
            print(f"🔒 Client {self.partition_id}: Training with STANDARD DP "
                  f"(noise={dp_config['noise_multiplier']:.2f}, clip={dp_config['max_grad_norm']:.2f})")

        else:
            # Compute per-round epsilon budget (divide total by num rounds)
            per_round_epsilon = 10.0 / max(self.total_rounds, 1)  # 10.0 / 10 = 1.0 per round
            base_noise = per_round_epsilon * (
                        1.0 - convergence_score * (self.current_round / max(self.total_rounds, 1)))
            base_noise = max(0.5, base_noise)

            # Get per-layer configuration
            dp_config = get_adaptive_dp_config(
                sensitivities=sensitivities,
                layer_classification=layer_classification,
                target_epsilon=base_noise,
                strategy="normalized"  # Respects epsilon budget
            )

            # Add metadata
            dp_config["per_layer"] = True
            dp_config["dataset_name"] = self.dataset_name

            dp_mode = "adaptive_dp_per_layer"

            print(f"🎯 Client {self.partition_id} [{device_profile.get('device_type', 'unknown')}]: "
                  f"Training with PER-LAYER ADAPTIVE DP")

            # Print sensitivity report (first round only)
            if self.current_round == 0:
                SensitivityReporter.print_sensitivity_report(
                    sensitivities,
                    layer_classification,
                    dp_config["noise_multipliers"]
                )

        try:
            results = train(
                model,
                self.trainloader,
                self.valloader,
                self.local_epochs,
                self.learning_rate,
                self.device,
                dp_config=dp_config,
                dataset_name=self.dataset_name,
            )

        except Exception as e:
            print(f"❌ Client {self.partition_id} training failed: {e}")
            import traceback
            traceback.print_exc()

            return get_weights(model), 0, {
                "train_loss": 0.0,
                "val_loss": 0.0,
                "val_accuracy": 0.0,
                "epsilon": 0.0,
                "error": str(e),
                "device_type": device_profile.get("device_type", "unknown"),
                "dp_mode": dp_mode,
                "dropped": 0,
            }

        metrics = {
            # Training metrics
            "train_loss": float(results.get("train_loss") or 0.0),
            "val_loss": float(results.get("val_loss") or 0.0),
            "val_accuracy": float(results.get("val_accuracy") or 0.0),

            # Privacy metrics
            "epsilon": float(results.get("epsilon") or 0.0),
            "dp_mode": dp_mode,
            "avg_sensitivity": float(avg_sensitivity),

            # Device type only (NO sensitive metrics like CPU%, RAM%, battery%)
            "device_type": str(device_profile.get("device_type", "unknown")),

            # DP config summary
            "per_layer_dp": bool(dp_config.get("per_layer", False)) if dp_config else False,
            "num_layers": len(sensitivities) if sensitivities else 0,

            # Status
            "dropped": 0,
        }

        return get_weights(model), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        """Evaluate global model on local test set."""
        model = make_model(self.dataset_name)
        model.to(self.device)
        set_weights(model, parameters)

        try:
            loss, accuracy = test(model, self.valloader, self.device, self.dataset_name)
        except Exception as e:
            print(f"❌ Client {self.partition_id} evaluation failed: {e}")
            return 0.0, len(self.valloader.dataset), {"accuracy": 0.0, "error": str(e)}

        return float(loss), len(self.valloader.dataset), {
            "accuracy": float(accuracy)
        }


def client_fn(context: Context):
    """
    Create client instance with configuration from context.
    
    Args:
        context: Flower context with node and run configuration
    
    Returns:
        ClientApp instance
    """

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    batch_size = int(context.run_config.get("batch-size", 32))
    dataset_name = str(context.run_config.get("dataset", "cifar10"))
    local_epochs = int(context.run_config.get("local-epochs", 2))
    learning_rate = float(context.run_config.get("learning-rate", 0.001))
    total_rounds = int(context.run_config.get("num-server-rounds", 10))
    device = str(context.run_config.get("device", "auto"))

    trainloader, valloader = load_data(
        partition_id,
        num_partitions,
        batch_size,
        dataset_name
    )

    return QPrivIoTClient(
        trainloader=trainloader,
        valloader=valloader,
        partition_id=partition_id,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        dataset_name=dataset_name,
        total_rounds=total_rounds,
        device=device,
    ).to_client()


app = ClientApp(
    client_fn=client_fn,
    mods=[secaggplus_mod],
)
