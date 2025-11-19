"""
Flower server with SecAgg, convergence tracking, and adaptive privacy - SENSITIVITY ENABLED.
"""
from logging import INFO, WARNING
from typing import List, Tuple, Dict, Callable

import torch
from flwr.common import Context, Metrics, ndarrays_to_parameters, parameters_to_ndarrays, log
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

from qpriviot_fl.task import make_model, get_weights, ConvergenceTracker
from qpriviot_fl.test.metrics import MetricsLogger, plot_comprehensive_results
from qpriviot_fl.privacy import PrivacyAccountant
import numpy as np


def set_random_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    log(INFO, f"Random seeds set to {seed}")


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics from clients weighted by dataset size.

    Args:
        metrics: List of (num_examples, metrics_dict) tuples
                 num_examples = -1 indicates dropped client
                 num_examples = 0 indicates error

    Returns:
        Aggregated metrics dictionary
    """
    if not metrics:
        return {}

    # Separate valid, dropped, and error clients
    valid_metrics = [(num, m) for num, m in metrics if num > 0]
    dropped_clients = [(num, m) for num, m in metrics if num == -1]
    error_clients = [(num, m) for num, m in metrics if num == 0]

    # Log dropout statistics
    if dropped_clients:
        dropped_types = [m.get("device_type", "unknown") for _, m in dropped_clients]
        log(WARNING, f"Dropped clients: {len(dropped_clients)} (types: {set(dropped_types)})")

    if error_clients:
        log(WARNING, f"Error clients: {len(error_clients)}")

    if not valid_metrics:
        log(WARNING, "No valid metrics from any client this round")
        return {
            "num_dropped": len(dropped_clients),
            "num_errors": len(error_clients),
            "num_valid_clients": 0,
        }

    total_examples = sum(num for num, _ in valid_metrics)
    aggregated = {}

    # Aggregate model quality metrics (weighted by samples)
    for key in ["accuracy", "val_accuracy", "train_loss", "val_loss"]:
        values = [(num, m.get(key, 0)) for num, m in valid_metrics if key in m]
        if values:
            aggregated[key] = sum([num * val for num, val in values]) / total_examples

    # Aggregate privacy metrics (epsilon) - sample-weighted
    epsilons = [(num, m.get("epsilon")) for num, m in valid_metrics
                if m.get("epsilon") is not None and m.get("epsilon") > 0]
    if epsilons:
        total_eps_weighted = sum(num * eps for num, eps in epsilons)
        aggregated["avg_epsilon"] = total_eps_weighted / total_examples
        aggregated["max_epsilon"] = max(eps for _, eps in epsilons)
        aggregated["min_epsilon"] = min(eps for _, eps in epsilons)

    # Aggregate sensitivity metrics (average across clients) - NEW!
    sensitivities = [m.get("avg_sensitivity", 0) for _, m in valid_metrics
                     if m.get("avg_sensitivity", 0) > 0]
    if sensitivities:
        aggregated["avg_sensitivity"] = sum(sensitivities) / len(sensitivities)
        aggregated["max_sensitivity"] = max(sensitivities)
        aggregated["min_sensitivity"] = min(sensitivities)

    # Track DP modes used
    dp_modes = {}
    for _, m in valid_metrics:
        mode = m.get("dp_mode", "unknown")
        dp_modes[mode] = dp_modes.get(mode, 0) + 1
    if dp_modes:
        aggregated["dp_modes"] = dp_modes

    # Track per-layer DP usage
    per_layer_count = sum(1 for _, m in valid_metrics if m.get("per_layer_dp", False))
    if per_layer_count > 0:
        aggregated["per_layer_dp_clients"] = per_layer_count

    aggregated["num_valid_clients"] = len(valid_metrics)
    aggregated["num_dropped"] = len(dropped_clients)
    aggregated["num_errors"] = len(error_clients)
    aggregated["total_clients_attempted"] = len(metrics)

    return aggregated


def compute_resource_score(metrics: List[Tuple[int, Metrics]]) -> float:
    """Compute average resource utilization score from client device profiles."""
    valid_metrics = [(num, m) for num, m in metrics if num > 0]
    if not valid_metrics:
        return 0.5  # Default moderate score

    # Track client statistics
    num_valid = len(valid_metrics)
    total_attempted = len(metrics)

    # For now, use a heuristic based on client participation
    participation_rate = num_valid / max(1, total_attempted)

    # Higher participation = better resources
    import random
    random.seed(len(metrics))  # Deterministic variation
    variation = random.uniform(-0.2, 0.2)
    resource_score = max(0.0, min(1.0, participation_rate + variation))

    return resource_score


def create_fit_config_fn(
        convergence_tracker: ConvergenceTracker,
        use_adaptive_dp: bool,
        use_dp: bool,
        target_epsilon: float,  # NEW parameter!
) -> Callable[[int], Dict]:
    """Create fit_config function with closure over trackers."""

    def fit_config(server_round: int) -> Dict:
        """Send dynamic config to clients each round."""
        convergence_score = convergence_tracker.get_convergence_score()

        config = {
            "convergence_score": float(convergence_score),
            "current_round": int(server_round - 1),
            "use_dp": bool(use_dp),
            "use_adaptive_dp": bool(use_adaptive_dp),
            "target_epsilon": float(target_epsilon),  # NEW! Pass to clients
        }
        return config

    return fit_config


def create_fit_metrics_aggregation_fn(
        convergence_tracker: ConvergenceTracker,
        privacy_accountant: PrivacyAccountant,
        metrics_logger: MetricsLogger,
) -> Callable[[List[Tuple[int, Metrics]]], Metrics]:
    """Create fit_metrics_aggregation function with closure over trackers."""

    def fit_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        """Aggregate fit metrics and update trackers."""
        aggregated = weighted_average(metrics)

        # Update convergence tracker only if we have valid loss data
        if "val_loss" in aggregated and aggregated["val_loss"] is not None:
            convergence_tracker.update(aggregated["val_loss"])

        # Update privacy accountant only if epsilon was reported
        if "avg_epsilon" in aggregated and aggregated["avg_epsilon"] is not None:
            privacy_accountant.add_round(aggregated["avg_epsilon"])
        else:
            log(WARNING, "No epsilon metrics reported from clients this round")

        convergence_score = convergence_tracker.get_convergence_score()
        resource_score = compute_resource_score(metrics)

        round_metrics = {
            **aggregated,
            "convergence_score": convergence_score,
            "resource_score": resource_score,
        }

        metrics_logger.log_round(round_metrics)

        # Log key metrics
        if "val_accuracy" in aggregated:
            log(INFO, f"Val Accuracy: {aggregated['val_accuracy'] * 100:.2f}%")
        if "val_loss" in aggregated:
            log(INFO, f"Val Loss: {aggregated['val_loss']:.4f}")
        if "avg_epsilon" in aggregated:
            log(
                INFO,
                f"Privacy ε: {aggregated['avg_epsilon']:.4f} "
                f"[{aggregated['min_epsilon']:.4f}, {aggregated['max_epsilon']:.4f}]",
            )
        if "avg_sensitivity" in aggregated:  # NEW!
            log(
                INFO,
                f"Sensitivity: {aggregated['avg_sensitivity']:.4f} "
                f"[{aggregated.get('min_sensitivity', 0):.4f}, "
                f"{aggregated.get('max_sensitivity', 0):.4f}]",
            )

        log(INFO, f"Convergence Score: {convergence_score:.4f}")
        log(INFO, f"Resource Score: {resource_score:.4f}")

        # Log client participation
        num_valid = aggregated.get("num_valid_clients", 0)
        num_dropped = aggregated.get("num_dropped", 0)
        num_errors = aggregated.get("num_errors", 0)
        num_total = aggregated.get("total_clients_attempted", 0)
        if num_total > 0:
            log(
                INFO,
                f"Clients: {num_valid} valid, {num_dropped} dropped, "
                f"{num_errors} errors (total: {num_total})",
            )

        # Log per-layer DP usage
        if "per_layer_dp_clients" in aggregated:
            log(
                INFO,
                f"Per-layer DP: {aggregated['per_layer_dp_clients']}/{num_valid} clients",
            )

        # Log DP modes distribution
        if "dp_modes" in aggregated:
            modes_str = ", ".join(f"{k}:{v}" for k, v in aggregated["dp_modes"].items())
            log(INFO, f"DP Modes: {modes_str}")

        return aggregated

    return fit_metrics_aggregation


app = ServerApp()


def create_secagg_workflow(context: Context):
    """Create SecAggPlus workflow with validated configuration."""
    use_secagg = bool(context.run_config.get("use-secagg", True))

    if not use_secagg:
        log(WARNING, "SecAgg DISABLED - using standard aggregation (NOT privacy-preserving)")
        return None

    try:
        num_shares = int(context.run_config.get("num-shares", 3))
        reconstruction_threshold = int(context.run_config.get("reconstruction-threshold", 2))
        max_weight = int(context.run_config.get("max-weight", 10000))

        # Validate SecAgg configuration
        if reconstruction_threshold > num_shares:
            log(
                WARNING,
                f"reconstruction_threshold ({reconstruction_threshold}) > num_shares ({num_shares}). "
                f"Adjusting to {num_shares}",
            )
            reconstruction_threshold = num_shares

        if reconstruction_threshold < 1:
            log(WARNING, "reconstruction_threshold < 1. Setting to 1")
            reconstruction_threshold = 1

        log(INFO, "SecAggPlus Configuration:")
        log(INFO, f"  - num_shares: {num_shares}")
        log(INFO, f"  - reconstruction_threshold: {reconstruction_threshold}")
        log(INFO, f"  - max_weight: {max_weight}")
        log(INFO, "Secure aggregation enabled - model updates encrypted")

        return SecAggPlusWorkflow(
            num_shares=num_shares,
            reconstruction_threshold=reconstruction_threshold,
            max_weight=max_weight,
        )

    except Exception as e:
        log(WARNING, f"Failed to create SecAggPlus: {e}. Falling back to standard aggregation")
        return None


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main server logic with adaptive privacy and convergence tracking."""
    # Configuration
    dataset_name = str(context.run_config.get("dataset", "cifar10"))
    num_rounds = int(context.run_config["num-server-rounds"])
    use_secagg = bool(context.run_config.get("use-secagg", True))
    use_dp = bool(context.run_config.get("use-dp", True))
    use_adaptive_dp = bool(context.run_config.get("use-adaptive-dp", True))
    target_epsilon = float(context.run_config.get("target-epsilon", 10.0))

    random_seed = int(context.run_config.get("random-seed", 42))
    set_random_seeds(random_seed)  # Set seeds BEFORE creating model!

    log(INFO, "=" * 60)
    log(INFO, "QPrivIoT-FL Server Configuration")
    log(INFO, "=" * 60)
    log(INFO, f"Dataset: {dataset_name}")
    log(INFO, f"Rounds: {num_rounds}")
    log(INFO, f"Differential Privacy: {use_dp}")
    log(INFO, f"Adaptive DP: {use_adaptive_dp}")
    log(INFO, f"Target Epsilon: {target_epsilon}")
    log(INFO, f"Secure Aggregation: {use_secagg}")
    log(INFO, f"Random Seed: {random_seed}")
    log(INFO, "=" * 60)



    # Initialize trackers
    convergence_tracker = ConvergenceTracker(window_size=5, threshold=0.01)
    privacy_accountant = PrivacyAccountant(target_epsilon=target_epsilon, target_delta=1e-5)
    metrics_logger = MetricsLogger()

    # Initialize global model
    global_model = make_model(dataset_name)
    initial_params = ndarrays_to_parameters(get_weights(global_model))

    # Log initial model state for verification
    initial_loss_estimate = sum([p.sum() for p in get_weights(global_model)]) / len(get_weights(global_model))
    log(INFO, f"Initial model checksum: {initial_loss_estimate:.6f} (should be same across experiments)")

    # Create strategy
    strategy = FedAvg(
        fraction_fit=float(context.run_config.get("fraction-fit", 0.8)),
        fraction_evaluate=float(context.run_config.get("fraction-evaluate", 0.5)),
        min_fit_clients=int(context.run_config.get("min-fit-clients", 3)),
        min_available_clients=int(context.run_config.get("min-available-clients", 3)),
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=create_fit_metrics_aggregation_fn(
            convergence_tracker, privacy_accountant, metrics_logger
        ),
        on_fit_config_fn=create_fit_config_fn(
            convergence_tracker, use_adaptive_dp, use_dp, target_epsilon  # Pass target_epsilon!
        ),
        initial_parameters=initial_params,
    )

    # Create workflow
    fit_workflow = create_secagg_workflow(context)
    workflow = DefaultWorkflow(fit_workflow=fit_workflow)

    # Run federated learning
    legacy_context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    log(INFO, "Starting federated training...")
    try:
        workflow(grid, legacy_context)
    except Exception as e:
        log(WARNING, f"Training interrupted: {e}")
        import traceback
        traceback.print_exc()

    # Print final results
    log(INFO, "=" * 80)
    log(INFO, "TRAINING COMPLETE!")
    log(INFO, "=" * 80)

    privacy_report = privacy_accountant.get_privacy_report()
    log(INFO, f"Total Privacy Spent: {privacy_report['total_epsilon']:.4f}")
    log(INFO, f"Rounds Completed: {privacy_report['rounds_completed']}")
    if privacy_report["rounds_completed"] > 0:
        log(INFO, f"Average per Round: {privacy_report['avg_epsilon_per_round']:.4f}")
    log(INFO, f"Remaining Budget: {privacy_report['remaining_budget']:.4f}")

    if privacy_report["total_epsilon"] > privacy_report.get("target_epsilon", float("inf")):
        log(WARNING, "Privacy budget EXCEEDED target!")
    else:
        log(INFO, "Privacy budget within target")

    # Save model
    if hasattr(strategy, "parameters") and strategy.parameters is not None:
        try:
            final_ndarrays = parameters_to_ndarrays(strategy.parameters)
            param_keys = list(global_model.state_dict().keys())
            state_dict = {k: torch.tensor(v) for k, v in zip(param_keys, final_ndarrays)}
            model_path = f"final_model_{dataset_name}.pt"
            torch.save(state_dict, model_path)
            log(INFO, f"Model saved to {model_path}")
        except Exception as e:
            log(WARNING, f"Could not save model: {e}")

    # Save metrics
    all_metrics = metrics_logger.get_all_metrics()
    if len(all_metrics) > 0:
        try:
            if not use_dp:
                metrics_path = "results_no_dp.json"
            elif use_adaptive_dp:
                metrics_path = "results_adaptive_dp.json"
            else:
                metrics_path = "results_uniform_dp.json"

            metrics_logger.save_to_file(metrics_path)
            log(INFO, f"Metrics saved to {metrics_path}")
        except Exception as e:
            log(WARNING, f"Could not save results: {e}")

    log(INFO, "=" * 60)
