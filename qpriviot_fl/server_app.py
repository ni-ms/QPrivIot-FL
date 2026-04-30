from __future__ import annotations

import json
import os
import math
import random
import secrets
import torch
from typing import List, Optional

import numpy as np
from flwr.common import (
    Context,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    FitIns,
    Parameters
)
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

from qpriviot_fl.privacy_utils import RenyiPrivacyAccountant, dequantize, SensitivityTracker, allocate_adaptive_noise
from qpriviot_fl.task import make_model, load_data


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
        
        self.loss_history = []
        self.convergence_score = 0.0
        self.accountant = RenyiPrivacyAccountant(target_epsilon=target_epsilon, target_delta=1e-5)
        self.sensitivity_tracker = SensitivityTracker(alpha=0.5)
        self.experiments_log = []
        self.secagg_seed = secrets.randbits(32)
        self._client_manager: ClientManager | None = None
        self.current_parameters: Parameters | None = None
        self.client_resources = {} # cid -> resource_score

    def _get_round_epsilon(self, server_round: int) -> float:
        """
        Calculates per-round epsilon using cosine annealing.
        References: Section 3.4
        """
        num_rounds = max(self.num_rounds, 1)
        # Calibration: To last 'num_rounds', each round should cost approx target_epsilon / num_rounds
        # The 2.0 multiplier is a calibration constant for Renyi Differential Privacy (RDP).
        # RDP composition theorems involve constants that can cause the privacy budget 
        # to be exhausted faster than a simple linear sqrt(T) division would suggest.
        # This buffer ensures the model reaches all T rounds for maximum utility.
        base_eps = self.target_epsilon / (math.sqrt(num_rounds) * 2.0)
        
        if not self.use_adaptive_dp:
            return base_eps

        epsilon_init = base_eps * 1.5
        epsilon_final = base_eps * 0.5
        
        # decay(t, T) = (1 + cos(pi * t / T)) / 2
        decay = (1 + math.cos(math.pi * (server_round - 1) / num_rounds)) / 2
        epsilon_t = epsilon_final + (epsilon_init - epsilon_final) * decay
        return epsilon_t

    def configure_fit(self, server_round, parameters, client_manager: ClientManager):
        self.current_parameters = parameters
        self._client_manager = client_manager

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

        # Get round budget and sensitivities
        epsilon_t = self._get_round_epsilon(server_round)
        sensitivities = self.sensitivity_tracker.get_sensitivities()

        dp_status = "Adaptive" if self.use_adaptive_dp else ("Fixed" if self.use_dp else "None")
        secagg_status = "Active" if self.use_secagg and len(clients) > 0 else "Off"
        print(f"\n--- Round {server_round} [DP: {dp_status}, SecAgg: {secagg_status}, Conv: {self.convergence_score:.2f}, ε_t: {epsilon_t:.3f}] ---")

        # Baseline learning rate passed during init
        base_lr = self.learning_rate
        # Apply 50% decay after reaching 25% of total rounds
        current_lr = base_lr * 0.5 if server_round > max(1, self.num_rounds * 0.25) else base_lr

        new_instructions = []
        total_clients = len(clients)
        secagg_quantization_bound = 10.0

        for idx, client_proxy in enumerate(clients):
            # Clamp sensitivities before JSON serialization to avoid division by zero from float underflow
            clamped_sensitivities = {k: max(v, 1e-6) for k, v in sensitivities.items()}

            config = {
                "use_secagg": self.use_secagg,
                "secagg_quantization_bound": secagg_quantization_bound,
                "use_dp": self.use_dp,
                "use_adaptive_dp": self.use_adaptive_dp,
                "convergence_score": self.convergence_score,
                "round": server_round,
                "epsilon_t": epsilon_t,
                "learning_rate": current_lr,
                "sensitivities": json.dumps(clamped_sensitivities)
            }

            if self.use_secagg:
                config["secagg_seed"] = self.secagg_seed + server_round
                config["secagg_client_index"] = idx
                config["secagg_total_clients"] = total_clients

            fit_ins = FitIns(parameters, config)
            new_instructions.append((client_proxy, fit_ins))

        return new_instructions

    def aggregate_fit(self, server_round, results, failures):
        """
        Aggregate fit results using resource-weighted FedAvg.
        NOTE: clients return DELTAS, not full weights. Do not call super().aggregate_fit()!
        """
        if not results:
            print("  No clients returned results. Skipping aggregation.")
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
            print(
                "  Error: Global model parameters (self.current_parameters) are not available. Cannot proceed with aggregation."
            )
            return None, {}

        current_parameters_ndarrays = parameters_to_ndarrays(self.current_parameters)

        is_secagg_round = results[0][1].metrics.get("secagg_active", False)

        averaged_delta = []
        if is_secagg_round:
            quantization_bound = results[0][1].metrics.get("secagg_quantization_bound", 10.0)
            print(f"  Aggregating Masked Updates (SecAgg: Summing Integers, bound: {quantization_bound})...")

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
            print("  Aggregating Deltas (Resource-Weighted FedAvg)...")
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
            self.accountant.add_round(noise_multiplier=accounting_noise, sampling_rate=sampling_rate)

        current_epsilon = self.accountant.get_total_epsilon()

        # Privacy budget enforcement: halt training if budget exceeded
        if dp_mode != "None" and current_epsilon > self.target_epsilon:
            print(f"  STOPPING: Privacy budget exceeded! ε={current_epsilon:.2f} > target={self.target_epsilon:.2f}")
            print(f"  Training halted to preserve differential privacy guarantees.")
            return None, {"budget_exceeded": True}

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
                current_norms[key] = torch.norm(grad_tensor).item()
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

        val_accuracy = np.mean([r.metrics.get("val_accuracy", 0.0) for _, r in results])
        val_loss = np.mean([r.metrics.get("val_loss", 0.0) for _, r in results])
        clip_norm = np.mean([r.metrics.get("clip_norm", 0.0) for _, r in results])
        avg_sensitivity = np.mean([r.metrics.get("avg_sensitivity", 0.0) for _, r in results])

        print(f"  Total Epsilon: {current_epsilon:.2f} | Avg. Loss: {avg_loss:.4f} | Val Acc: {val_accuracy:.4f}")

        # Get per-round epsilon and sensitivities for plotting
        epsilon_t_current = self._get_round_epsilon(server_round) if hasattr(self, '_get_round_epsilon') else 0.0
        current_sensitivities = self.sensitivity_tracker.get_sensitivities() if hasattr(self, 'sensitivity_tracker') else {}

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

        return agg_params, {}

    def _save(self, aggregated_model_ndarrays: List[np.ndarray]):
        """Saves the results log and the aggregated model parameters."""

        with open(self.results_file, "w") as f:
            json.dump({"rounds": self.experiments_log}, f, indent=2)

        print(f"  Saving global model to {self.model_file}...")
        np.savez(self.model_file, *aggregated_model_ndarrays)


def server_fn(context: Context):
    """Server function - returns ServerAppComponents."""

    num_rounds = context.run_config.get("num-server-rounds", 10)
    dataset = context.run_config.get("dataset", "cifar10")

    use_secagg = context.run_config.get("use-secagg", False)
    use_dp = context.run_config.get("use-dp", False)
    use_adaptive_dp = context.run_config.get("use-adaptive-dp", False)

    filename_parts = ["results"]

    if use_adaptive_dp:
        filename_parts.append("adaptive_dp")
    elif use_dp:
        filename_parts.append("uniform_dp")
    else:
        filename_parts.append("no_dp")

    if use_secagg:
        filename_parts.append("secagg")

    results_file_name = "_".join(filename_parts) + ".json"

    model_file_name = context.run_config.get("model-output-file", "global_model.npz")

    model = make_model(dataset)

    init_parameters_ndarrays = [v.detach().cpu().numpy() for _, v in model.named_parameters()]
    init_parameters = ndarrays_to_parameters(init_parameters_ndarrays)

    strategy = ProgressivePrivacyStrategy(
        results_file=results_file_name,
        model_file=model_file_name,
        use_secagg=use_secagg,
        use_dp=use_dp,
        use_adaptive_dp=use_adaptive_dp,
        dataset=dataset,
        fraction_fit=context.run_config.get("fraction-fit", 1.0),
        fraction_evaluate=context.run_config.get("fraction-evaluate", 1.0),
        min_fit_clients=context.run_config.get("min-fit-clients", 1),
        min_available_clients=context.run_config.get("min-available-clients", 1),
        learning_rate=context.run_config.get("learning-rate", 0.01),
        target_epsilon=context.run_config.get("target-epsilon", 3.0),
        num_rounds=num_rounds,
        initial_parameters=init_parameters,
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )


app = ServerApp(server_fn=server_fn)
