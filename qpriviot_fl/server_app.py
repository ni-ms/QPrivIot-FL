"""Flower server with SecAgg, convergence tracking, and adaptive privacy - FIXED."""

from logging import INFO, WARNING
from typing import List, Tuple, Dict, Callable
import json

import torch
from flwr.common import Context, Metrics, ndarrays_to_parameters, parameters_to_ndarrays, log
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

from qpriviot_fl.task import make_model, get_weights, ConvergenceTracker
from qpriviot_fl.metrics import MetricsLogger, plot_comprehensive_results
from qpriviot_fl.privacy import PrivacyAccountant


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate metrics from clients (weighted by dataset size).
    
    Args:
        metrics: List of (num_examples, metrics_dict) tuples
    
    Returns:
        Aggregated metrics dictionary
    """
    if not metrics:
        return {}

    valid_metrics = [(num, m) for num, m in metrics if num > 0]

    if not valid_metrics:
        return {}

    total_examples = sum([num for num, _ in valid_metrics])
    aggregated = {}

    for key in ["accuracy", "val_accuracy", "train_loss", "val_loss"]:
        values = [num * m.get(key, 0) for num, m in valid_metrics if key in m]
        if values:
            aggregated[key] = sum(values) / total_examples

    for key in ["resource_score", "cpu_percent", "ram_percent", "battery_percent", "bandwidth_mbps"]:
        values = [m.get(key, 0) for _, m in valid_metrics if key in m]
        if values:
            aggregated[key] = sum(values) / len(values)

    epsilons = [m.get("epsilon") for _, m in valid_metrics if m.get("epsilon") is not None and m.get("epsilon") > 0]
    if epsilons:
        aggregated["avg_epsilon"] = sum(epsilons) / len(epsilons)
        aggregated["max_epsilon"] = max(epsilons)
        aggregated["min_epsilon"] = min(epsilons)

    sensitivities = [m.get("avg_sensitivity", 0) for _, m in valid_metrics if m.get("avg_sensitivity", 0) > 0]
    if sensitivities:
        aggregated["avg_sensitivity"] = sum(sensitivities) / len(sensitivities)

    num_dropped = sum(1 for _, m in metrics if m.get("dropped", 0) == 1)
    if num_dropped > 0:
        aggregated["num_dropped"] = num_dropped

    return aggregated


def create_fit_config_fn(
        convergence_tracker: ConvergenceTracker,
        use_adaptive_dp: bool,
        use_dp: bool,
) -> Callable[[int], Dict]:
    """
    Create fit_config function with closure over trackers.
    ✅ FIXED: Proper closure to access trackers
    """

    def fit_config(server_round: int) -> Dict:
        """Send dynamic config to clients each round."""
        convergence_score = convergence_tracker.get_convergence_score()

        config = {
            "convergence_score": float(convergence_score),
            "current_round": int(server_round - 1),
            "use_dp": bool(use_dp),
            "use_adaptive_dp": bool(use_adaptive_dp),
        }

        return config

    return fit_config


def create_fit_metrics_aggregation_fn(
        convergence_tracker: ConvergenceTracker,
        privacy_accountant: PrivacyAccountant,
        metrics_logger: MetricsLogger,
) -> Callable[[List[Tuple[int, Metrics]]], Metrics]:
    """
    Create fit_metrics_aggregation function with closure over trackers.
    ✅ FIXED: Proper closure
    """

    def fit_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        """Aggregate fit metrics and update trackers."""
        aggregated = weighted_average(metrics)

        if "val_loss" in aggregated:
            convergence_tracker.update(aggregated["val_loss"])

        if "avg_epsilon" in aggregated:
            privacy_accountant.add_round(aggregated["avg_epsilon"])

        round_metrics = {
            "convergence_score": convergence_tracker.get_convergence_score(),
            **aggregated
        }

        metrics_logger.log_round(round_metrics)

        log(INFO, f"Val Accuracy: {aggregated.get('val_accuracy', 0) * 100:.2f}%")
        log(INFO, f"Val Loss: {aggregated.get('val_loss', 0):.4f}")

        if "avg_epsilon" in aggregated:
            log(INFO, f"Avg Privacy (ε): {aggregated['avg_epsilon']:.4f}")

        log(INFO, f"Convergence Score: {convergence_tracker.get_convergence_score():.4f}")

        if aggregated.get("num_dropped", 0) > 0:
            log(WARNING, f"Dropped Clients: {aggregated['num_dropped']}")

        return aggregated

    return fit_metrics_aggregation


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Main server logic with adaptive privacy and convergence tracking.
    ✅ FIXED: Creates fresh instances per run (thread-safe)
    """

    dataset_name = str(context.run_config.get("dataset", "cifar10"))
    num_rounds = int(context.run_config["num-server-rounds"])
    use_secagg = bool(context.run_config.get("use-secagg", True))
    use_dp = bool(context.run_config.get("use-dp", True))
    use_adaptive_dp = bool(context.run_config.get("use-adaptive-dp", True))
    target_epsilon = float(context.run_config.get("target-epsilon", 10.0))

    log(INFO, f"\n{'=' * 60}")
    log(INFO, "QPrivIoT-FL Server Configuration")
    log(INFO, f"{'=' * 60}")
    log(INFO, f"Dataset: {dataset_name}")
    log(INFO, f"Rounds: {num_rounds}")
    log(INFO, f"Differential Privacy: {use_dp}")
    log(INFO, f"Adaptive DP: {use_adaptive_dp}")
    log(INFO, f"Target Epsilon: {target_epsilon}")
    log(INFO, f"Secure Aggregation: {use_secagg}")
    log(INFO, f"{'=' * 60}\n")

    convergence_tracker = ConvergenceTracker(window_size=5, threshold=0.01)
    privacy_accountant = PrivacyAccountant(target_epsilon=target_epsilon, target_delta=1e-5)
    metrics_logger = MetricsLogger()

    global_model = make_model(dataset_name)
    initial_params = ndarrays_to_parameters(get_weights(global_model))

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
            convergence_tracker, use_adaptive_dp, use_dp
        ),
        initial_parameters=initial_params,
    )

    if use_secagg:
        fit_workflow = SecAggPlusWorkflow(
            num_shares=int(context.run_config.get("num-shares", 3)),
            reconstruction_threshold=int(context.run_config.get("reconstruction-threshold", 2)),
            max_weight=int(context.run_config.get("max-weight", 10000)),
        )
        log(INFO, "Secure aggregation commencing.")
    else:
        fit_workflow = None
        log(WARNING, "SecAgg disabled - using standard aggregation")

    workflow = DefaultWorkflow(fit_workflow=fit_workflow)

    legacy_context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    log(INFO, f"Starting federated training...")

    try:
        workflow(grid, legacy_context)
    except Exception as e:
        log(WARNING, f"Training interrupted: {e}")
        import traceback
        traceback.print_exc()

    log(INFO, f"\n{'=' * 60}")
    log(INFO, "Training Complete!")
    log(INFO, f"{'=' * 60}")

    privacy_report = privacy_accountant.get_privacy_report()
    log(INFO, f"Total Privacy Spent (ε): {privacy_report['total_epsilon']:.4f}")
    log(INFO, f"Rounds Completed: {privacy_report['rounds_completed']}")

    if privacy_report['rounds_completed'] > 0:
        log(INFO, f"Average ε per Round: {privacy_report['avg_epsilon_per_round']:.4f}")
        log(INFO, f"Remaining Budget: {privacy_report['remaining_budget']:.4f}")

    if hasattr(strategy, 'parameters') and strategy.parameters is not None:
        try:
            final_ndarrays = parameters_to_ndarrays(strategy.parameters)
            param_keys = list(global_model.state_dict().keys())
            state_dict = {k: torch.tensor(v) for k, v in zip(param_keys, final_ndarrays)}

            model_path = f"final_model_{dataset_name}.pt"
            torch.save(state_dict, model_path)
            log(INFO, f"Model saved to {model_path}")
        except Exception as e:
            log(WARNING, f"Could not save model: {e}")

    all_metrics = metrics_logger.get_all_metrics()
    if len(all_metrics) > 0:
        try:
            metrics_path = f"training_metrics_{dataset_name}.json"
            metrics_logger.save_to_file(metrics_path)

            plot_path = f"results_{dataset_name}.png"
            plot_comprehensive_results(all_metrics, privacy_report, save_path=plot_path)

            log(INFO, f"Metrics saved to {metrics_path}")
            log(INFO, f"Results plot saved to {plot_path}")
        except Exception as e:
            log(WARNING, f"Could not save results: {e}")

    log(INFO, f"{'=' * 60}\n")
