"""QPrivIot-FL: A Flower / PyTorch app."""
from collections import OrderedDict

import torch
from flwr.server import ServerApp
from flwr.serverapp.strategy import FedAvg  # Import from flwr.serverapp.strategy, NOT flwr.server.strategy
from flwr.server.grid import Grid
from flwr.common import ArrayRecord, ConfigRecord, Context

from qpriviot_fl.task import Net

app = ServerApp()

# Progressive privacy scheduler parameters
INITIAL_PRIVACY_BUDGET = 2.0
MIN_PRIVACY_BUDGET = 0.5
DECAY_RATE = 0.9


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train = context.run_config.get("fraction-train", 0.5)
    num_rounds = context.run_config.get("num-server-rounds", 10)
    lr = context.run_config.get("lr", 0.001)

    # Load global model
    global_model = Net()
    # Get state_dict as OrderedDict (it already is one, but make it explicit)
    state_dict = global_model.state_dict()

    # Create ArrayRecord with OrderedDict[str, Tensor]
    arrays = ArrayRecord(OrderedDict(state_dict))

    # Initialize FedAvg strategy with adaptive client selection
    strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=1.0,
        min_available_nodes=2,
    )

    privacy_budget = INITIAL_PRIVACY_BUDGET

    for round_num in range(num_rounds):
        print(f"\n[Round {round_num + 1}/{num_rounds}] Privacy budget: {privacy_budget:.3f}")

        # Create train config with current privacy budget and learning rate
        train_config = ConfigRecord({"lr": lr, "privacy_budget": privacy_budget})

        # Run one round of federated learning
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=train_config,
            num_rounds=1,
        )

        # Update global model with aggregated result
        arrays = result.arrays

        # Progressive privacy budget decay
        privacy_budget = max(MIN_PRIVACY_BUDGET, privacy_budget * DECAY_RATE)

    # Save final model to disk
    print("\nSaving final model...")
    state_dict = arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
    print("Training complete!")
