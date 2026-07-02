from __future__ import annotations

import json
import os
import math
import random
import secrets
import torch
import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

from opacus.accountants.utils import get_noise_multiplier

import numpy as np
from flwr.common import (
    Context,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    FitIns,
    Parameters,
    Metrics,
    Scalar
)
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

from qpriviot_fl.privacy_utils import RenyiPrivacyAccountant, dequantize, SensitivityTracker, allocate_adaptive_noise
from qpriviot_fl.task import make_model, load_data, test


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregates metrics by weighting them by the number of examples."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    accuracies = [num_examples * m["val_accuracy"] for num_examples, m in metrics if "val_accuracy" in m]
    losses = [num_examples * m["val_loss"] for num_examples, m in metrics if "val_loss" in m]

    result: Metrics = {}
    if accuracies and total_examples > 0:
        result["val_accuracy"] = sum(accuracies) / total_examples
    if losses and total_examples > 0:
        result["val_loss"] = sum(losses) / total_examples
    return result


class ProgressivePrivacyStrategy(FedAvg):
    def __init__(self,
                 results_file="results.json",
                 model_file="global_model.npz",
                 use_secagg=False,
                 use_dp=False,
                 use_adaptive_dp=False,
                 target_epsilon=3.0,
                 num_rounds=100,
                 learning_rate=0.01,
                 dataset="cifar10",
                 seed=42,
                 alpha=0.3,
                 num_clients=10,
                 ablation_mode: str = "",
                 secagg_dp_mode: str = "",
                 **kwargs):
        super().__init__(**kwargs)
        self.results_file = os.path.abspath(results_file)
        self.model_file = os.path.abspath(model_file)
        self.use_secagg = use_secagg
        self.use_dp = use_dp
        self.use_adaptive_dp = use_adaptive_dp
        self.target_epsilon = target_epsilon
        self.num_rounds = num_rounds
        self.learning_rate = learning_rate
        self.dataset = dataset
        self.seed = seed
        self.alpha = alpha
        self.num_clients = num_clients
        self.ablation_mode = ablation_mode
        self.secagg_dp_mode = secagg_dp_mode

        self.target_delta = 1e-5

        # Early stopping state
        self._best_val_loss = float('inf')
        self._patience = 0
        self._early_stop = False

        self.loss_history = []
        self.convergence_score = 0.0

        # Section 4.1: Calibrate noise multiplier using Opacus
        if use_dp or use_adaptive_dp:
            # Opacus expects 'epochs' to be the total number of training steps.
            # In FL, T rounds with 1 local epoch per round = T total epochs on the dataset
            # (assuming we sample fraction_fit clients each round).
            self.sigma = get_noise_multiplier(
                target_epsilon=target_epsilon,
                target_delta=self.target_delta,
                sample_rate=self.fraction_fit,
                epochs=self.num_rounds
            )
            _log(f"  [DP CALIBRATION] Target ε={target_epsilon}, T={num_rounds}, q={self.fraction_fit} => σ={self.sigma:.4f}")
        else:
            self.sigma = 0.0

        self.accountant = RenyiPrivacyAccountant(target_epsilon=target_epsilon, target_delta=self.target_delta)
        self.sensitivity_tracker = SensitivityTracker(alpha=0.5)
        self.experiments_log = []
        self.per_client_log = []
        self.secagg_seed = secrets.randbits(32)
        self._client_manager: ClientManager | None = None
        self.current_parameters: Parameters | None = None
        self.client_resources = {} # cid -> resource_score
        
        # Load test data for centralized evaluation
        _, self.test_loader = load_data(0, 1, 32, dataset, alpha=100.0, seed=seed)

    def _get_round_epsilon(self, server_round: int) -> float:
        base_eps = self.target_epsilon / self.num_rounds

        if not self.use_adaptive_dp:
            return base_eps

        # param-only ablation: no round schedule — use flat per-round budget
        if self.ablation_mode == "param-only":
            return base_eps

        # Full AdaPriv and round-only: cosine annealing over training
        epsilon_init = base_eps * 1.5
        epsilon_final = base_eps * 0.5
        decay = (1 + math.cos(math.pi * (server_round - 1) / self.num_rounds)) / 2
        return epsilon_final + (epsilon_init - epsilon_final) * decay

    def configure_fit(self, server_round, parameters, client_manager: ClientManager):
        self.current_parameters = parameters
        self._client_manager = client_manager

        # Log current privacy budget (do not halt — Opacus already calibrated sigma for
        # num_rounds total steps, so running all rounds stays within the privacy guarantee)
        current_epsilon = self.accountant.get_total_epsilon()
        if (self.use_dp or self.use_adaptive_dp) and current_epsilon > 0:
            _log(f"  Privacy budget: ε_spent={current_epsilon:.3f} / ε_target={self.target_epsilon:.2f}")

        # Resource-aware sampling for AdaPriv: prioritize devices that have reported high scores
        if self.use_adaptive_dp and self.client_resources:
            all_clients = client_manager.all()
            available_cids = list(all_clients.keys())
            
            # Sort by remembered resource score (fallback to 0.5)
            sorted_cids = sorted(available_cids, 
                               key=lambda cid: self.client_resources.get(cid, 0.5), 
                               reverse=True)
            
            # Sample top 70% high-resource for stability, 30% random for fairness/diversity
            num_to_sample = int(client_manager.num_available() * self.fraction_fit)
            num_to_sample = max(num_to_sample, self.min_fit_clients)
            
            num_high = int(num_to_sample * 0.7)
            num_rand = num_to_sample - num_high
            
            selected_cids = sorted_cids[:num_high]
            remaining_cids = sorted_cids[num_high:]
            
            if num_rand > 0 and remaining_cids:
                import random
                selected_cids.extend(random.sample(remaining_cids, min(num_rand, len(remaining_cids))))
            
            clients = [all_clients[cid] for cid in selected_cids]
        elif self.use_secagg:
            num_available = client_manager.num_available()
            num_to_sample = max(int(num_available * self.fraction_fit), self.min_fit_clients)
            clients = client_manager.sample(
                num_clients=num_to_sample,
                min_num_clients=self.min_fit_clients
            )
        else:
            num_available = client_manager.num_available()
            num_to_sample = max(int(num_available * self.fraction_fit), self.min_fit_clients)
            clients = client_manager.sample(
                num_clients=num_to_sample,
                min_num_clients=self.min_fit_clients
            )

        if not clients:
            return []

        # Update convergence score using training loss instead of metrics to avoid oscillation
        if len(self.loss_history) >= 2:
            recent = self.loss_history[-3:]
            mean_loss = float(np.mean(recent))
            if mean_loss > 0:
                cv = float(np.std(recent)) / mean_loss
                self.convergence_score = max(0.0, min(1.0, 1.0 - (cv * 4)))

        # Get round budget and sensitivities.
        # Warmup period: don't send sensitivity estimates for the first few rounds.
        # Round-1 sensitivity estimates are noise-dominated (no signal history yet) and
        # cause destructive per-layer allocations that spike the training loss.
        # Scale warmup with num_rounds: 5% of rounds (min 5, max 10) for a full window.
        epsilon_t = self._get_round_epsilon(server_round)
        sensitivity_warmup = max(5, min(10, self.num_rounds // 20))
        if self.use_adaptive_dp and server_round <= sensitivity_warmup:
            sensitivities = {}  # uniform noise during warmup (like fixed-DP)
        else:
            sensitivities = self.sensitivity_tracker.get_sensitivities()

        dp_status = "Adaptive" if self.use_adaptive_dp else ("Fixed" if self.use_dp else "None")
        secagg_status = "Active" if self.use_secagg and len(clients) > 0 else "Off"
        _log(f"\n--- Round {server_round} [DP: {dp_status}, SecAgg: {secagg_status}, Conv: {self.convergence_score:.2f}, ε_t: {epsilon_t:.3f}] ---")

        # Baseline learning rate passed during init
        base_lr = self.learning_rate
        
        # Section 4.4: Cosine learning rate schedule with warmup
        warmup_rounds = 5
        if server_round <= warmup_rounds:
            current_lr = base_lr * server_round / warmup_rounds
        else:
            progress = (server_round - warmup_rounds) / max(1, self.num_rounds - warmup_rounds)
            current_lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))

        new_instructions = []
        total_clients = len(clients)
        secagg_quantization_bound = 10.0

        for idx, client_proxy in enumerate(clients):
            # Clamp sensitivities before JSON serialization to avoid division by zero from float underflow
            clamped_sensitivities = {k: max(v, 1e-6) for k, v in sensitivities.items()}

            config = {
                "use_secagg": self.use_secagg,
                "secagg_quantization_bound": secagg_quantization_bound,
                "secagg_dp_mode": self.secagg_dp_mode,
                "use_dp": self.use_dp,
                "use_adaptive_dp": self.use_adaptive_dp,
                "ablation_mode": self.ablation_mode,
                "convergence_score": self.convergence_score,
                "round": server_round,
                "epsilon_t": epsilon_t,
                "sigma": self.sigma,
                "learning_rate": current_lr,
                "proximal_mu": 0.01 if not self.use_dp and not self.use_adaptive_dp else 0.0,
                "dirichlet_alpha": self.alpha,
                "seed": self.seed,
                "num_clients": self.num_clients,
                "num_rounds": self.num_rounds,
                "target_epsilon": self.target_epsilon,
                "avg_resource_score": float(np.mean(list(self.client_resources.values()))) if self.client_resources else 0.3,
                "sensitivities": json.dumps(clamped_sensitivities)
            }

            if self.use_secagg:
                config["secagg_seed"] = self.secagg_seed + server_round
                config["secagg_client_index"] = idx
                config["secagg_total_clients"] = total_clients

            fit_ins = FitIns(parameters, config)
            new_instructions.append((client_proxy, fit_ins))

        return new_instructions

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Centralized evaluation on the server."""
        if self.test_loader is None:
            return None

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        model = make_model(self.dataset).to(device)
        
        # Set model parameters
        params_ndarray = parameters_to_ndarrays(parameters)
        for (_, param), new_val in zip(model.named_parameters(), params_ndarray):
            tensor = torch.from_numpy(new_val).to(device=device, dtype=param.dtype)
            param.data = tensor.view(param.shape)
            
        loss, accuracy = test(model, self.test_loader, device)
        _log(f"  [CENTRALIZED EVAL] Round {server_round}: loss={loss:.4f}, accuracy={accuracy:.4f}")
        return loss, {"accuracy": accuracy}

    def aggregate_fit(self, server_round, results, failures):
        """
        Aggregate fit results using resource-weighted FedAvg.
        NOTE: clients return DELTAS, not full weights. Do not call super().aggregate_fit()!
        """
        if not results:
            _log("  No clients returned results. Skipping aggregation.")
            return None, {}

        # Update local knowledge of client resources
        for proxy, res in results:
            score = res.metrics.get("resource_score", 0.5)
            self.client_resources[proxy.cid] = score
            # print(f"DEBUG: Cid {proxy.cid} reported resource {score:.2f}")

        total_available_clients = 1
        if self._client_manager:
            total_available_clients = max(self._client_manager.num_available(), len(results))
        else:
            total_available_clients = max(1, len(results))
            
        if self.use_adaptive_dp:
            avg_res = np.mean(list(self.client_resources.values())) if self.client_resources else 0
            # print(f"DEBUG: Round {server_round} aggregated. Known resources: {len(self.client_resources)}, Avg: {avg_res:.2f}")

        if self.current_parameters is None:
            _log("  Error: Global model parameters (self.current_parameters) are not available. Cannot proceed with aggregation.")
            return None, {}

        current_parameters_ndarrays = parameters_to_ndarrays(self.current_parameters)

        is_secagg_round = results[0][1].metrics.get("secagg_active", False)

        averaged_delta = []
        if is_secagg_round:
            quantization_bound = results[0][1].metrics.get("secagg_quantization_bound", 10.0)
            _log(f"  Aggregating Masked Updates (SecAgg: Summing Integers, bound: {quantization_bound})...")

            first_res_params = parameters_to_ndarrays(results[0][1].parameters)
            aggregated_integers_delta = [
                arr.astype(np.int64).copy() for arr in first_res_params
            ]

            for _, res in results[1:]:
                masked_delta_params = parameters_to_ndarrays(res.parameters)
                for i in range(len(masked_delta_params)):
                    aggregated_integers_delta[i] += masked_delta_params[i].astype(np.int64)

            final_floats_delta = dequantize(aggregated_integers_delta, clip_range=quantization_bound, range_max=1000000)

            num_participants = len(results)
            averaged_delta = [
                d / num_participants for d in final_floats_delta
            ]

        else:
            _log("  Aggregating Deltas (Resource-Weighted FedAvg)...")
            # In AdaPriv, we weight updates by both sample count and the reported readiness score
            # This incentivizes participation from high-resource devices
            if self.use_adaptive_dp:
                total_weight = sum(max(res.num_examples * res.metrics.get("resource_score", 1.0), 1e-6) for _, res in results)
            else:
                total_weight = sum(res.num_examples for _, res in results)
            
            # Initialize with zeros
            first_deltas = parameters_to_ndarrays(results[0][1].parameters)
            averaged_delta = [np.zeros_like(d) for d in first_deltas]
            
            for _, res in results:
                if self.use_adaptive_dp:
                    client_weight = max(res.num_examples * res.metrics.get("resource_score", 1.0), 1e-6)
                else:
                    client_weight = res.num_examples
                
                weight = client_weight / total_weight if total_weight > 0 else 1.0 / len(results)
                client_deltas = parameters_to_ndarrays(res.parameters)
                for i in range(len(client_deltas)):
                    averaged_delta[i] += client_deltas[i] * weight

        # Use min noise across participants for formal DP guarantees (worst-case)
        noise_values = [r.metrics.get("avg_noise", 0) for _, r in results]
        accounting_noise = float(np.min(noise_values)) if noise_values else 0.0
        sampling_rate = len(results) / total_available_clients
        dp_mode = results[0][1].metrics.get("dp_mode", "None")
        
        if dp_mode != "None" and accounting_noise > 0:
            # Opacus uses steps=1 because we pass sample_rate and epochs to get_noise_multiplier
            # which already accounts for multiple steps/epochs if calibrated that way.
            # In Flower, each round is essentially one step in the global accountant's view.
            self.accountant.add_round(noise_multiplier=accounting_noise, sampling_rate=sampling_rate, steps=1)

        current_epsilon = self.accountant.get_total_epsilon()

        # Update sensitivity tracker with averaged deltas as proxies for global gradients
        model_keys = list(self.sensitivity_tracker.history.keys())
        if not model_keys:
             # Initialize keys using the actual configured dataset
             init_model = make_model(self.dataset)
             model_keys = list(init_model.state_dict().keys())
        
        current_norms = {}
        current_grads = {}
        for i, key in enumerate(model_keys):
            if i < len(averaged_delta):
                grad_tensor = torch.from_numpy(averaged_delta[i])
                # Normalize by sqrt(num_params) → per-parameter RMS magnitude.
                # Without this, fc1 (262k params) always dominates conv1 (288 params)
                # purely from dimensionality, not actual data sensitivity.
                n_params = max(grad_tensor.numel() ** 0.5, 1.0)
                current_norms[key] = torch.norm(grad_tensor).item() / n_params
                current_grads[key] = grad_tensor
        
        self.sensitivity_tracker.update(current_norms, current_grads)

        new_global_params_ndarrays = [
            current_parameters_ndarrays[i] + averaged_delta[i]
            for i in range(len(current_parameters_ndarrays))
        ]
        agg_params = ndarrays_to_parameters(new_global_params_ndarrays)

        losses = [r.metrics["train_loss"] for _, r in results if "train_loss" in r.metrics]
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        self.loss_history.append(avg_loss)

        # Weighted aggregation of metrics for logging
        total_n = sum(res.num_examples for _, res in results)
        if total_n > 0:
            val_accuracy = sum(res.metrics.get("val_accuracy", 0.0) * res.num_examples for _, res in results) / total_n
            val_loss = sum(res.metrics.get("val_loss", 0.0) * res.num_examples for _, res in results) / total_n
        else:
            val_accuracy = np.mean([r.metrics.get("val_accuracy", 0.0) for _, r in results])
            val_loss = np.mean([r.metrics.get("val_loss", 0.0) for _, r in results])

        clip_norm = np.mean([r.metrics.get("clip_norm", 0.0) for _, r in results])
        avg_sensitivity = np.mean([r.metrics.get("avg_sensitivity", 0.0) for _, r in results])

        # Section 4.4: Early stopping on val_loss.
        # For DP modes, val_loss is noisy so use a larger patience proportional to
        # num_rounds. Minimum 10 rounds without improvement for DP, 5 for no-DP.
        if self.use_dp or self.use_adaptive_dp:
            patience_limit = max(15, self.num_rounds // 5)
        else:
            patience_limit = max(10, self.num_rounds // 4)
        if val_loss < self._best_val_loss - 1e-3:
            self._best_val_loss = val_loss
            self._patience = 0
        else:
            self._patience += 1
            if self._patience >= patience_limit:
                _log(f"  EARLY STOPPING: No improvement in val_loss for {patience_limit} rounds.")
                self._early_stop = True

        _log(f"  Total Epsilon: {current_epsilon:.2f} | Avg. Loss: {avg_loss:.4f} | Val Acc: {val_accuracy:.4f}")

        # Get per-round epsilon and sensitivities for plotting
        epsilon_t_current = self._get_round_epsilon(server_round) if hasattr(self, '_get_round_epsilon') else 0.0
        current_sensitivities = self.sensitivity_tracker.get_sensitivities() if hasattr(self, 'sensitivity_tracker') else {}

        # Log per-client results (includes fairness metrics for device-tier analysis)
        for proxy, res in results:
            self.per_client_log.append({
                "round": server_round,
                "client_id": proxy.cid,
                "val_accuracy": res.metrics.get("val_accuracy", 0.0),
                "val_loss": res.metrics.get("val_loss", 0.0),
                "train_loss": res.metrics.get("train_loss", 0.0),
                "num_examples": res.num_examples,
                "resource_score": res.metrics.get("resource_score", 0.0),
                "device_id": res.metrics.get("device_id", 0),
                "dp_mode": res.metrics.get("dp_mode", "None"),
                "avg_noise": res.metrics.get("avg_noise", 0.0),
                "clip_norm": res.metrics.get("clip_norm", 0.0),
            })

        record = {
            "round": server_round,
            "avg_loss": avg_loss,
            "val_accuracy": val_accuracy,
            "val_loss": val_loss,
            "convergence_score": self.convergence_score,
            "total_epsilon": current_epsilon,
            "epsilon_t": epsilon_t_current,  # Per-round epsilon for Graph 5
            "sensitivities": dict(current_sensitivities),  # Per-layer sensitivities for Graph 6
            "avg_latency": np.mean([r.metrics.get("client_latency", 0) for _, r in results]),
            "avg_resource": np.mean([r.metrics.get("resource_score", 0) for _, r in results]),
            "clip_norm": clip_norm,
            "avg_sensitivity": avg_sensitivity,
            "secagg_active": is_secagg_round,
            "dp_mode": dp_mode,
        }
        self.experiments_log.append(record)
        self._save(new_global_params_ndarrays)

        if self._early_stop:
            return None, {"early_stop": True}

        return agg_params, {}

    def _save(self, aggregated_model_ndarrays: List[np.ndarray]):
        """Saves the results log and the aggregated model parameters."""

        with open(self.results_file, "w") as f:
            json.dump({"rounds": self.experiments_log}, f, indent=2)

        if self.per_client_log:
            pd.DataFrame(self.per_client_log).to_csv(
                self.results_file.replace(".json", "_per_client.csv"), index=False
            )

        _log(f"  Saving global model to {self.model_file}...")
        np.savez(self.model_file, *aggregated_model_ndarrays)


def server_fn(context: Context):
    """Server function - returns ServerAppComponents."""

    num_rounds = context.run_config.get("num-server-rounds", 10)
    dataset = context.run_config.get("dataset", "cifar10")

    use_secagg = context.run_config.get("use-secagg", False)
    use_dp = context.run_config.get("use-dp", False)
    use_adaptive_dp = context.run_config.get("use-adaptive-dp", False)
    ablation_mode = str(context.run_config.get("ablation-mode", ""))
    secagg_dp_mode = str(context.run_config.get("secagg-dp-mode", ""))

    seed = context.run_config.get("seed", 42)
    target_epsilon = context.run_config.get("target-epsilon", 3.0)

    filename_parts = ["results", dataset]

    if ablation_mode:
        filename_parts.append(ablation_mode)
    elif use_adaptive_dp:
        filename_parts.append("adapriv")
    elif use_dp:
        filename_parts.append("fixed-dp")
    else:
        filename_parts.append("no-dp")

    if use_secagg:
        filename_parts.append("secagg")
    if secagg_dp_mode:
        filename_parts.append(secagg_dp_mode)

    # Disambiguate scale-sweep runs by client count for ALL configs (no-dp / fixed-dp /
    # secagg) so non-secagg controls don't overwrite each other across N. Legacy N=10
    # filenames are left unchanged.
    n_clients = context.run_config.get("num-clients", 10)
    if n_clients != 10:
        filename_parts.append(f"n{n_clients}")

    filename_parts.append(f"seed{seed}")
    filename_parts.append(f"eps{target_epsilon}")

    results_file_name = os.path.join("experiment_results", "_".join(filename_parts) + ".json")

    model_file_name = context.run_config.get("model-output-file", "global_model.npz")

    # Seed model init so all configs (no-dp / fixed-dp / adapriv) start from the same weights.
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(dataset)

    init_parameters_ndarrays = [v.detach().cpu().numpy() for _, v in model.named_parameters()]
    init_parameters = ndarrays_to_parameters(init_parameters_ndarrays)

    strategy = ProgressivePrivacyStrategy(
        results_file=results_file_name,
        model_file=model_file_name,
        use_secagg=use_secagg,
        use_dp=use_dp,
        use_adaptive_dp=use_adaptive_dp,
        ablation_mode=ablation_mode,
        secagg_dp_mode=secagg_dp_mode,
        dataset=dataset,
        seed=seed,
        alpha=context.run_config.get("dirichlet-alpha", 0.3),
        num_clients=context.run_config.get("num-clients", 10),
        fraction_fit=context.run_config.get("fraction-fit", 1.0),
        fraction_evaluate=context.run_config.get("fraction-evaluate", 1.0),
        min_fit_clients=context.run_config.get("min-fit-clients", 1),
        min_available_clients=context.run_config.get("min-available-clients", 1),
        learning_rate=context.run_config.get("learning-rate", 0.01),
        target_epsilon=target_epsilon,
        num_rounds=num_rounds,
        initial_parameters=init_parameters,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )


app = ServerApp(server_fn=server_fn)
