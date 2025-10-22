"""QPrivIot-FL: Enhanced server with convergence-aware privacy scheduling and resource-based client selection."""
from collections import OrderedDict
import torch
from flwr.server import ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.server.grid import Grid
from flwr.common import ArrayRecord, ConfigRecord, Context, MetricRecord
import numpy as np

from qpriviot_fl.task import Net, FEMNISTNet

app = ServerApp()

INITIAL_PRIVACY_BUDGET = 2.0
MIN_PRIVACY_BUDGET = 0.5
DECAY_RATE = 0.9


class AdaptivePrivacyScheduler:
    """Convergence-aware privacy budget scheduler."""

    def __init__(self, initial_budget, min_budget, decay_rate):
        self.current_budget = initial_budget
        self.min_budget = min_budget
        self.decay_rate = decay_rate
        self.loss_history = []
        self.acc_history = []

    def update(self, round_num, avg_loss, avg_acc):
        """Update privacy budget based on convergence metrics."""
        self.loss_history.append(avg_loss)
        self.acc_history.append(avg_acc)

        if len(self.loss_history) >= 3:
            recent_loss_change = abs(self.loss_history[-1] - self.loss_history[-2])

            if recent_loss_change < 0.01:

                adaptive_decay = self.decay_rate * 0.85
            elif recent_loss_change > 0.1:

                adaptive_decay = self.decay_rate * 1.1
            else:

                adaptive_decay = self.decay_rate
        else:

            adaptive_decay = self.decay_rate

        self.current_budget = max(self.min_budget, self.current_budget * adaptive_decay)

        return self.current_budget

    def get_budget(self):
        """Get current privacy budget."""
        return self.current_budget


class ResourceAwareClientSelector:
    """Select clients based on resource availability."""

    def __init__(self, min_battery=20, min_resource_score=0.3):
        self.min_battery = min_battery
        self.min_resource_score = min_resource_score
        self.client_metrics = {}

    def update_client_metrics(self, client_metrics_list):
        """Store client resource metrics from last round."""
        for metrics in client_metrics_list:
            client_id = metrics.get("client_id", "unknown")
            self.client_metrics[client_id] = {
                "battery_percent": metrics.get("battery_percent", 100),
                "resource_score": metrics.get("resource_score", 0.5),
                "energy_consumed": metrics.get("energy_consumed", 0),
            }

    def is_client_eligible(self, client_id):
        """Check if client meets minimum resource requirements."""
        if client_id not in self.client_metrics:
            return True

        metrics = self.client_metrics[client_id]
        battery_ok = metrics["battery_percent"] >= self.min_battery
        resource_ok = metrics["resource_score"] >= self.min_resource_score

        return battery_ok and resource_ok

    def get_client_priority(self, client_id):
        """Calculate client priority score for selection."""
        if client_id not in self.client_metrics:
            return 0.5

        metrics = self.client_metrics[client_id]

        priority = (
                metrics["battery_percent"] / 100 * 0.5 +
                metrics["resource_score"] * 0.5
        )

        return priority


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp with adaptive features."""

    fraction_train = context.run_config.get("fraction-train", 0.5)
    num_rounds = context.run_config.get("num-server-rounds", 10)
    lr = context.run_config.get("lr", 0.001)
    dataset_name = context.run_config.get("dataset", "cifar10")

    if dataset_name == "cifar10":
        global_model = Net()
    elif dataset_name == "femnist":
        global_model = FEMNISTNet()
    else:
        global_model = Net()

    state_dict = global_model.state_dict()
    arrays = ArrayRecord(OrderedDict(state_dict))

    privacy_scheduler = AdaptivePrivacyScheduler(
        INITIAL_PRIVACY_BUDGET, MIN_PRIVACY_BUDGET, DECAY_RATE
    )
    client_selector = ResourceAwareClientSelector(min_battery=20, min_resource_score=0.3)

    strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=1.0,
        min_available_nodes=2,
    )

    round_metrics = {
        "round": [],
        "privacy_budget": [],
        "avg_loss": [],
        "avg_accuracy": [],
        "avg_epsilon": [],
        "avg_energy": [],
        "num_clients": [],
    }

    for round_num in range(num_rounds):
        current_budget = privacy_scheduler.get_budget()

        print(f"\n{'=' * 60}")
        print(f"Round {round_num + 1}/{num_rounds}")
        print(f"Privacy Budget: {current_budget:.4f}")
        print(f"{'=' * 60}")

        train_config = ConfigRecord({
            "lr": lr,
            "privacy_budget": current_budget
        })

        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=train_config,
            num_rounds=1,
        )

        arrays = result.arrays

        if hasattr(result, 'metrics') and result.metrics:
            metrics_data = result.metrics

            avg_loss = metrics_data.get("train_loss", 0.0)
            avg_acc = metrics_data.get("train_acc", 0.0)
            avg_epsilon = metrics_data.get("dp_epsilon", 0.0)
            avg_energy = metrics_data.get("energy_consumed", 0.0)
            num_clients = metrics_data.get("num_clients", 0)

            new_budget = privacy_scheduler.update(round_num, avg_loss, avg_acc)

            round_metrics["round"].append(round_num + 1)
            round_metrics["privacy_budget"].append(current_budget)
            round_metrics["avg_loss"].append(avg_loss)
            round_metrics["avg_accuracy"].append(avg_acc)
            round_metrics["avg_epsilon"].append(avg_epsilon)
            round_metrics["avg_energy"].append(avg_energy)
            round_metrics["num_clients"].append(num_clients)

            print(f"Avg Train Loss: {avg_loss:.4f}")
            print(f"Avg Train Accuracy: {avg_acc:.4f}")
            print(f"Avg Epsilon: {avg_epsilon:.4f}")
            print(f"Avg Energy Consumed: {avg_energy:.4f}%")
            print(f"Next Round Budget: {new_budget:.4f}")

    print(f"\n{'=' * 60}")
    print("Training Complete! Saving final model...")
    print(f"{'=' * 60}")

    state_dict = arrays.to_torch_state_dict()
    torch.save(state_dict, f"final_model_{dataset_name}.pt")

    np.savez(
        f"training_metrics_{dataset_name}.npz",
        **round_metrics
    )

    print(f"\nModel saved to: final_model_{dataset_name}.pt")
    print(f"Metrics saved to: training_metrics_{dataset_name}.npz")

    print(f"\n{'=' * 60}")
    print("Training Summary:")
    print(f"{'=' * 60}")
    print(f"Final Loss: {round_metrics['avg_loss'][-1]:.4f}")
    print(f"Final Accuracy: {round_metrics['avg_accuracy'][-1]:.4f}")
    print(f"Total Privacy Spent (avg epsilon): {round_metrics['avg_epsilon'][-1]:.4f}")
    print(f"Total Energy Consumed (avg): {sum(round_metrics['avg_energy']):.2f}%")
