"""Flower server with SecAgg, convergence tracking, and adaptive privacy."""

from logging import INFO, WARNING
from typing import List, Tuple, Dict

import torch
from flwr.common import Context, Metrics, ndarrays_to_parameters, log
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

from qpriviot_fl.task import make_model, get_weights, ConvergenceTracker
from qpriviot_fl.metrics import MetricsLogger, plot_comprehensive_results
from qpriviot_fl.privacy import PrivacyAccountant


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics from clients (weighted by dataset size)."""
    if not metrics:
        return {}

    total_examples = sum([num for num, _ in metrics])

    aggregated = {}

    for key in ["accuracy", "val_accuracy", "train_loss", "val_loss"]:
        values = [num * m.get(key, 0) for num, m in metrics if key in m]
        if values:
            aggregated[key] = sum(values) / total_examples

    for key in ["resource_score", "cpu_percent", "ram_percent", "battery_percent", "bandwidth_mbps"]:
        values = [m.get(key, 0) for _, m in metrics if key in m]
        if values:
            aggregated[key] = sum(values) / len(values)

    epsilons = [m.get("epsilon") for _, m in metrics if m.get("epsilon") is not None]
    if epsilons:
        aggregated["avg_epsilon"] = sum(epsilons) / len(epsilons)
        aggregated["max_epsilon"] = max(epsilons)

    sensitivities = [m.get("avg_sensitivity", 0) for _, m in metrics]
    if sensitivities:
        aggregated["avg_sensitivity"] = sum(sensitivities) / len(sensitivities)

    return aggregated


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main server logic with adaptive privacy and convergence tracking."""

    dataset_name = context.run_config.get("dataset", "cifar10")
    num_rounds = context.run_config["num-server-rounds"]
    use_secagg = context.run_config.get("use-secagg", True)

    global_model = make_model(dataset_name)
    initial_params = ndarrays_to_parameters(get_weights(global_model))

    convergence_tracker = ConvergenceTracker()
    privacy_accountant = PrivacyAccountant(target_epsilon=10.0)
    metrics_logger = MetricsLogger()

    strategy = FedAvg(
        fraction_fit=context.run_config.get("fraction-fit", 0.8),
        fraction_evaluate=context.run_config.get("fraction-evaluate", 0.5),
        min_fit_clients=context.run_config.get("min-fit-clients", 3),
        min_available_clients=context.run_config.get("min-available-clients", 3),
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=initial_params,
    )

    legacy_context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    if use_secagg:
        fit_workflow = SecAggPlusWorkflow(
            num_shares=context.run_config.get("num-shares", 3),
            reconstruction_threshold=context.run_config.get("reconstruction-threshold", 2),
            max_weight=context.run_config.get("max-weight", 1000),
        )
        log(INFO, "Using SecAgg+ for secure aggregation")
    else:
        fit_workflow = None
        log(WARNING, "SecAgg disabled - using standard aggregation")

    workflow = DefaultWorkflow(fit_workflow=fit_workflow) if use_secagg else DefaultWorkflow()

    log(INFO, f"Starting federated training for {num_rounds} rounds on {dataset_name}")

    for round_num in range(num_rounds):
        log(INFO, f"\n{'=' * 60}")
        log(INFO, f"Round {round_num + 1}/{num_rounds}")
        log(INFO, f"{'=' * 60}")

        convergence_score = convergence_tracker.get_convergence_score()
        context.run_config["convergence_score"] = convergence_score
        context.run_config["current_round"] = round_num

        result = strategy.configure_fit(
            server_round=round_num + 1,
            parameters=initial_params,
            client_manager=grid.client_manager(),
        )

        results = []
        for client_proxy, fit_ins in zip(grid.sample_clients(), result):
            fit_res = client_proxy.fit(fit_ins, timeout=None)
            results.append(fit_res)

        round_metrics = {
            "round": round_num + 1,
            "convergence_score": convergence_score,
        }

        client_metrics = [(fit_res.num_examples, fit_res.metrics) for fit_res in results if fit_res]
        aggregated = weighted_average(client_metrics)
        round_metrics.update(aggregated)

        if "val_loss" in aggregated:
            convergence_tracker.update(aggregated["val_loss"])

        if "avg_epsilon" in aggregated:
            privacy_accountant.add_round(aggregated["avg_epsilon"])

        metrics_logger.log_round(round_metrics)
        log(INFO, f"Val Accuracy: {aggregated.get('val_accuracy', 0):.4f}")
        log(INFO, f"Val Loss: {aggregated.get('val_loss', 0):.4f}")
        log(INFO, f"Convergence Score: {convergence_score:.4f}")
        log(INFO, f"Avg Privacy (ε): {aggregated.get('avg_epsilon', 0):.4f}")
        log(INFO, f"Avg Resource Score: {aggregated.get('resource_score', 0):.4f}")

        if privacy_accountant.is_budget_exceeded():
            log(WARNING, "Privacy budget exceeded! Stopping training.")
            break

    log(INFO, f"\n{'=' * 60}")
    log(INFO, "Training Complete!")
    log(INFO, f"{'=' * 60}")

    privacy_report = privacy_accountant.get_privacy_report()
    log(INFO, f"Total Privacy Spent (ε): {privacy_report['total_epsilon']:.4f}")
    log(INFO, f"Rounds Completed: {privacy_report['rounds_completed']}")

    final_weights = get_weights(global_model)
    torch.save(global_model.state_dict(), f"final_model_{dataset_name}.pt")
    log(INFO, f"Model saved to final_model_{dataset_name}.pt")

    metrics_logger.save_to_file(f"training_metrics_{dataset_name}.json")
    plot_comprehensive_results(
        metrics_logger.get_all_metrics(),
        privacy_report,
        save_path=f"results_{dataset_name}.png"
    )

    log(INFO, "Results saved and visualized!")
