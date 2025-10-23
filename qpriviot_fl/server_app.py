"""QPrivIot-FL: Fixed server with proper ArrayRecord creation."""
from collections import OrderedDict
import torch
from flwr.server import ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.server.grid import Grid
from flwr.common import ArrayRecord, ConfigRecord, Context

from qpriviot_fl.task import Net, FEMNISTNet

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Simplified working main entry point."""

    num_rounds = context.run_config.get("num-server-rounds", 10)
    lr = context.run_config.get("lr", 0.001)
    dataset_name = context.run_config.get("dataset", "cifar10")
    privacy_budget = 2.0

    if dataset_name == "femnist":
        model = FEMNISTNet()
    else:
        model = Net()

    state_dict = OrderedDict(model.state_dict())

    print(f"\nInitializing model for {dataset_name}")
    print(f"State dict contains {len(state_dict)} parameters")

    if len(state_dict) == 0:
        raise ValueError("Model state_dict is empty! Check model definition.")

    initial_arrays = ArrayRecord(state_dict)

    strategy = FedAvg(
        fraction_train=0.5,
        fraction_evaluate=1.0,
        min_available_nodes=2,
    )

    train_config = ConfigRecord({
        "lr": lr,
        "privacy_budget": privacy_budget
    })

    print(f"\n{'=' * 60}")
    print(f"Starting Federated Learning")
    print(f"Dataset: {dataset_name}")
    print(f"Rounds: {num_rounds}")
    print(f"Privacy Budget: {privacy_budget}")
    print(f"{'=' * 60}\n")

    try:
        result = strategy.start(
            grid=grid,
            initial_arrays=initial_arrays,
            train_config=train_config,
            num_rounds=num_rounds,
        )

        print(f"\n{'=' * 60}")
        print("Training Complete!")
        print(f"{'=' * 60}\n")

        if result.arrays and len(result.arrays) > 0:
            final_state_dict = result.arrays.to_torch_state_dict()
            torch.save(final_state_dict, f"final_model_{dataset_name}.pt")
            print(f"✓ Model saved: final_model_{dataset_name}.pt")
            print(f"  Contains {len(final_state_dict)} parameters")
        else:
            print("⚠ Warning: No model received or empty arrays")

        if result.metrics:
            print(f"\nFinal Metrics:")
            for key, value in result.metrics.items():
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
