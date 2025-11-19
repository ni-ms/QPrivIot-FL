"""
Flower client with SecAgg, adaptive DP, and device profiling - SENSITIVITY ENABLED.
"""
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context

from qpriviot_fl.device_profile import (
    profile_device,
    is_eligible_for_training,
    get_device_aware_noise_multiplier,
)
from qpriviot_fl.sensitivity import (
    compute_adaptive_sensitivity,
    get_adaptive_dp_noise_config,
)
from qpriviot_fl.task import (
    make_model,
    get_weights,
    set_weights,
    load_data,
    train,
    test,
    ConvergenceTracker,
)


class QPrivIoTClient(NumPyClient):
    """FL client with adaptive privacy and resource awareness.

    Features:
    - Device profiling for heterogeneous IoT environments
    - Sensitivity-based noise allocation (NOW ENABLED!)
    - Convergence-aware privacy scheduling
    - Secure aggregation with SecAgg+
    """

    def __init__(
            self,
            train_loader,
            val_loader,
            partition_id: int,
            local_epochs: int,
            learning_rate: float,
            dataset_name: str,
            total_rounds: int,
            device: str = "auto",
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.partition_id = partition_id
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.dataset_name = dataset_name
        self.total_rounds = total_rounds
        self.current_round = 0
        self.last_valid_round = -1
        self.convergence_tracker = ConvergenceTracker(window_size=3)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def fit(self, parameters, config):
        """Train model with configurable DP modes.

        Supports:
        - No DP (baseline)
        - Standard DP (fixed noise)
        - Adaptive DP (sensitivity-based per-layer noise) ← NOW WORKS!

        Returns:
            Tuple of (weights, num_examples, metrics)
        """
        # Profile device
        device_profile = profile_device()

        # Round counter with fallback
        config_round = int(config.get("current_round", -1))
        if config_round >= 0:
            self.current_round = config_round
            self.last_valid_round = config_round
        elif self.last_valid_round >= 0:
            self.current_round = self.last_valid_round + 1
            print(f"Client {self.partition_id}: Using round fallback {self.current_round}")
        else:
            print(f"Client {self.partition_id}: No valid round received")

        convergence_score = float(config.get("convergence_score", 0.0))
        use_dp = bool(config.get("use_dp", True))
        use_adaptive_dp = bool(config.get("use_adaptive_dp", True))
        target_epsilon = float(config.get("target_epsilon", 1.0))

        # Check device eligibility
        if not is_eligible_for_training(device_profile):
            print(
                f"Client {self.partition_id} ({device_profile.get('device_type', 'unknown')}) "
                f"dropped (resource_score={device_profile['resource_score']:.2f})"
            )
            model = make_model(self.dataset_name)
            model.to(self.device)
            set_weights(model, parameters)
            return (
                get_weights(model),
                -1,
                {
                    "train_loss": 0.0,
                    "val_loss": 0.0,
                    "val_accuracy": 0.0,
                    "epsilon": 0.0,
                    "device_type": device_profile.get("device_type", "unknown"),
                    "avg_sensitivity": 0.0,
                    "noise_multiplier": 0.0,
                    "max_grad_norm": 0.0,
                    "dropped": 1,
                },
            )

        # Create model ONCE before any checks
        model = make_model(self.dataset_name)
        model.to(self.device)
        set_weights(model, parameters)

        avg_sensitivity = 0.0

        if not use_dp:
            dp_config = None
            dp_mode = "no_dp"
            print(f"Client {self.partition_id}: Training WITHOUT DP")

        elif not use_adaptive_dp:
            # Standard DP (global noise)
            dp_config = {
                "noise_multiplier": 1.0,
                "max_grad_norm": 1.0,
                "per_layer": False,
            }
            dp_mode = "standard_dp"
            print(
                f"Client {self.partition_id}: Training with STANDARD DP "
                f"(noise={dp_config['noise_multiplier']:.2f}, "
                f"clip={dp_config['max_grad_norm']:.2f})"
            )

        else:
            print(f"Client {self.partition_id}: Computing per-layer sensitivities...")

            try:
                # Step 1: Compute sensitivities
                normalized_sens, raw_sens = compute_adaptive_sensitivity(
                    model=model,
                    dataloader=self.train_loader,
                    num_batches=3
                )

                print(f"  → Computed sensitivities for {len(normalized_sens)} layers")
                avg_sensitivity = sum(normalized_sens.values()) / max(len(normalized_sens), 1)
                print(f"  → Average sensitivity: {avg_sensitivity:.4f}")

                # Step 2: Get adaptive noise configuration
                dp_config = get_adaptive_dp_noise_config(
                    sensitivities=normalized_sens,
                    target_epsilon=target_epsilon,
                    delta=1e-5,
                    clip_norm=1.0,
                    strategy="gaussian"
                )



                dp_mode = "adaptive_sensitivity_dp"

                print(f"Client {self.partition_id}: ADAPTIVE SENSITIVITY DP enabled")
                print(f"  → Target epsilon: {target_epsilon:.2f}")
                print(f"  → Computed epsilon: {dp_config.get('total_epsilon', 0):.2f}")
                print(f"  → Per-layer noise: {len(dp_config['noise_multipliers'])} layers")


            except Exception as e:
                print(f"Client {self.partition_id}: Sensitivity computation failed: {e}")
                import traceback
                traceback.print_exc()

                # Fallback to convergence-aware DP
                base_noise = get_device_aware_noise_multiplier(
                    device_profile, convergence_score
                )
                convergence_reduction = (1.0 - convergence_score) * 0.4
                noise_multiplier = base_noise * (1.0 - convergence_reduction)
                noise_multiplier = max(noise_multiplier, 0.5)

                dp_config = {
                    "noise_multiplier": noise_multiplier,
                    "max_grad_norm": 1.0,
                    "per_layer": False,
                    "device_type": device_profile.get("device_type", "unknown"),
                    "resource_score": device_profile.get("resource_score", 0.5),
                    "convergence_score": convergence_score,
                }
                dp_mode = "convergence_aware_dp_fallback"
                print(f"  → Using fallback convergence-aware DP (noise={noise_multiplier:.2f})")

        # Train the model
        try:
            results = train(
                model,
                self.train_loader,
                self.val_loader,
                self.local_epochs,
                self.learning_rate,
                self.device,
                dp_config=dp_config,
                dataset_name=self.dataset_name,
            )

            train_loss = results.get("train_loss", 0.0)
            if train_loss > 0:
                self.convergence_tracker.update(train_loss)

        except Exception as e:
            print(f"Client {self.partition_id} training failed: {e}")
            import traceback
            traceback.print_exc()
            return (
                get_weights(model),
                0,
                {
                    "train_loss": 0.0,
                    "val_loss": 0.0,
                    "val_accuracy": 0.0,
                    "epsilon": 0.0,
                    "error": str(e),
                    "device_type": device_profile.get("device_type", "unknown"),
                    "dp_mode": dp_mode,
                    "dropped": 0,
                },
            )

        # Prepare metrics
        metrics = {
            "train_loss": float(results.get("train_loss") or 0.0),
            "val_loss": float(results.get("val_loss") or 0.0),
            "val_accuracy": float(results.get("val_accuracy") or 0.0),
            "epsilon": float(results.get("epsilon") or 0.0),
            "dp_mode": dp_mode,
            "device_type": str(device_profile.get("device_type", "unknown")),
            "resource_score": float(device_profile.get("resource_score", 0.0)),
            "avg_sensitivity": float(avg_sensitivity),  # NEW!
            "convergence_score": float(convergence_score),
            "dropped": 0,
        }

        # Add per-layer DP metrics if applicable
        if dp_config and dp_config.get("per_layer", False):
            metrics["per_layer_dp"] = True
            metrics["num_layers"] = len(dp_config.get("noise_multipliers", {}))
            if "total_epsilon" in dp_config:
                metrics["total_epsilon"] = float(dp_config["total_epsilon"])
        else:
            metrics["noise_multiplier"] = float(dp_config.get("noise_multiplier", 0.0) if dp_config else 0.0)

        return get_weights(model), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        """Evaluate global model on local test set."""
        model = make_model(self.dataset_name)
        model.to(self.device)
        set_weights(model, parameters)

        try:
            loss, accuracy = test(model, self.val_loader, self.device, self.dataset_name)
        except Exception as e:
            print(f"Client {self.partition_id} evaluation failed: {e}")
            return 0.0, len(self.val_loader.dataset), {"accuracy": 0.0, "error": str(e)}

        return float(loss), len(self.val_loader.dataset), {"accuracy": float(accuracy)}


def client_fn(context: Context):
    """Create client instance with configuration from context.

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

    train_loader, val_loader = load_data(
        partition_id, num_partitions, batch_size, dataset_name
    )

    return QPrivIoTClient(
        train_loader=train_loader,
        val_loader=val_loader,
        partition_id=partition_id,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        dataset_name=dataset_name,
        total_rounds=total_rounds,
        device=device,
    ).to_client()


app = ClientApp(client_fn=client_fn, mods=[secaggplus_mod])
