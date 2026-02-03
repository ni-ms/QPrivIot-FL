from __future__ import annotations

import json
import os
import math
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
                 **kwargs):
        super().__init__(**kwargs)
        self.results_file = os.path.abspath(results_file)
        self.model_file = os.path.abspath(model_file)
        self.use_secagg = use_secagg
        self.use_dp = use_dp
        self.use_adaptive_dp = use_adaptive_dp
        self.target_epsilon = target_epsilon
        self.num_rounds = num_rounds
        
        self.loss_history = []
        self.convergence_score = 0.0
        self.accountant = RenyiPrivacyAccountant(target_epsilon=target_epsilon, target_delta=1e-5)
        self.sensitivity_tracker = SensitivityTracker(alpha=0.5)
        self.experiments_log = []
        self.secagg_seed = 1000
        self._client_manager: ClientManager | None = None
        self.current_parameters: Parameters | None = None

    def _get_round_epsilon(self, server_round: int) -> float:
        """
        Calculates per-round epsilon using cosine annealing.
        References: Section 3.4
        """
        if not self.use_adaptive_dp:
            return self.target_epsilon / math.sqrt(self.num_rounds)

        epsilon_init = (self.target_epsilon / math.sqrt(self.num_rounds)) * 1.5
        epsilon_final = (self.target_epsilon / math.sqrt(self.num_rounds)) * 0.5
        
        # decay(t, T) = (1 + cos(pi * t / T)) / 2
        decay = (1 + math.cos(math.pi * (server_round - 1) / self.num_rounds)) / 2
        epsilon_t = epsilon_final + (epsilon_init - epsilon_final) * decay
        return epsilon_t

    def configure_fit(self, server_round, parameters, client_manager: ClientManager):
        self.current_parameters = parameters
        self._client_manager = client_manager

        if self.use_secagg:
            clients = client_manager.sample(
                num_clients=client_manager.num_available(),
                min_num_clients=1
            )
        else:
            client_instructions = super().configure_fit(server_round, parameters, client_manager)
            clients = [instr[0] for instr in client_instructions]

        if not clients:
            return []

        # Update convergence score
        if len(self.loss_history) >= 2:
            recent = self.loss_history[-3:]
            mean_loss = np.mean(recent)
            if mean_loss > 0:
                cv = np.std(recent).item() / mean_loss.item()
                self.convergence_score = max(0.0, min(1.0, 1.0 - (cv * 4)))

        # Get round budget and sensitivities
        epsilon_t = self._get_round_epsilon(server_round)
        sensitivities = self.sensitivity_tracker.get_sensitivities()

        dp_status = "Adaptive" if self.use_adaptive_dp else ("Fixed" if self.use_dp else "None")
        secagg_status = "Active" if self.use_secagg and len(clients) > 0 else "Off"
        print(f"\n--- Round {server_round} [DP: {dp_status}, SecAgg: {secagg_status}, Conv: {self.convergence_score:.2f}, ε_t: {epsilon_t:.3f}] ---")

        new_instructions = []
        total_clients = len(clients)

        for idx, client_proxy in enumerate(clients):
            config = {
                "use_secagg": self.use_secagg,
                "use_dp": self.use_dp,
                "use_adaptive_dp": self.use_adaptive_dp,
                "convergence_score": self.convergence_score,
                "round": server_round,
                "epsilon_t": epsilon_t,
                "sensitivities": json.dumps(sensitivities)
            }

            if self.use_secagg:
                config["secagg_seed"] = self.secagg_seed + server_round
                config["secagg_client_index"] = idx
                config["secagg_total_clients"] = total_clients

            fit_ins = FitIns(parameters, config)
            new_instructions.append((client_proxy, fit_ins))

        return new_instructions

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            print("  No clients returned results. Skipping aggregation.")
            return None, {}

        total_available_clients = 1
        if self._client_manager:
            total_available_clients = self._client_manager.num_available()

        if self.current_parameters is None:
            print(
                "  Error: Global model parameters (self.current_parameters) are not available. Cannot proceed with aggregation."
            )
            return None, {}

        current_parameters_ndarrays = parameters_to_ndarrays(self.current_parameters)

        is_secagg_round = results[0][1].metrics.get("secagg_active", False)

        averaged_delta = []
        if is_secagg_round:
            print("  Aggregating Masked Updates (SecAgg: Summing Integers)...")

            first_res_params = parameters_to_ndarrays(results[0][1].parameters)
            aggregated_integers_delta = [
                arr.astype(np.int64).copy() for arr in first_res_params
            ]

            for _, res in results[1:]:
                masked_delta_params = parameters_to_ndarrays(res.parameters)
                for i in range(len(masked_delta_params)):
                    aggregated_integers_delta[i] += masked_delta_params[i].astype(np.int64)

            final_floats_delta = dequantize(aggregated_integers_delta, clip_range=1.0, range_max=1000000)

            num_participants = len(results)
            averaged_delta = [
                d / num_participants for d in final_floats_delta
            ]

        else:
            print("  Aggregating Deltas (Standard FedAvg)...")
            num_participants = len(results)
            deltas = parameters_to_ndarrays(results[0][1].parameters)

            for _, res in results[1:]:
                client_deltas = parameters_to_ndarrays(res.parameters)
                for i in range(len(deltas)):
                    deltas[i] += client_deltas[i]

            averaged_delta = [d / num_participants for d in deltas]

        # Update sensitivity tracker with averaged deltas as proxies for global gradients
        model_keys = self.sensitivity_tracker.history.keys()
        if not model_keys:
             # Initialize keys from first result if empty
             mock_model = make_model("cifar10") # Placeholder to get keys
             model_keys = list(mock_model.state_dict().keys())
        
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

        avg_noise = np.mean([r.metrics.get("avg_noise", 0) for _, r in results])
        sampling_rate = len(results) / total_available_clients
        dp_mode = results[0][1].metrics.get("dp_mode", "None")

        if dp_mode != "None":
            self.accountant.add_round(noise_multiplier=float(avg_noise), sampling_rate=sampling_rate)

        current_epsilon = self.accountant.get_total_epsilon()

        losses = [r.metrics["train_loss"] for _, r in results if "train_loss" in r.metrics]
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        self.loss_history.append(avg_loss)

        val_accuracy = np.mean([r.metrics.get("val_accuracy", 0.0) for _, r in results])
        val_loss = np.mean([r.metrics.get("val_loss", 0.0) for _, r in results])
        clip_norm = np.mean([r.metrics.get("clip_norm", 0.0) for _, r in results])
        avg_sensitivity = np.mean([r.metrics.get("avg_sensitivity", 0.0) for _, r in results])

        print(f"  Total Epsilon: {current_epsilon:.2f} | Avg. Loss: {avg_loss:.4f} | Val Acc: {val_accuracy:.4f}")

        record = {
            "round": server_round,
            "avg_loss": avg_loss,
            "val_accuracy": val_accuracy,
            "val_loss": val_loss,
            "convergence_score": self.convergence_score,
            "total_epsilon": current_epsilon,
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

    init_parameters_ndarrays = [v.detach().cpu().numpy() for v in model.state_dict().values()]
    init_parameters = ndarrays_to_parameters(init_parameters_ndarrays)

    strategy = ProgressivePrivacyStrategy(
        results_file=results_file_name,
        model_file=model_file_name,
        use_secagg=use_secagg,
        use_dp=use_dp,
        use_adaptive_dp=use_adaptive_dp,
        fraction_fit=context.run_config.get("fraction-fit", 1.0),
        fraction_evaluate=context.run_config.get("fraction-evaluate", 1.0),
        min_fit_clients=context.run_config.get("min-fit-clients", 1),
        min_available_clients=context.run_config.get("min-available-clients", 1),
        initial_parameters=init_parameters,
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )


app = ServerApp(server_fn=server_fn)
