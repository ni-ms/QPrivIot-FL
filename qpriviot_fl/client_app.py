import math
import time
import torch
import numpy as np
import traceback
from flwr.client import NumPyClient
from flwr.clientapp import ClientApp
from flwr.common import Context
import json

from qpriviot_fl.config import DEFAULT_CONFIG as CONFIG
from qpriviot_fl.device_profile import profile_device
from qpriviot_fl.task import load_data, make_model, train, test, get_weights, set_weights
from qpriviot_fl.privacy_utils import (
    allocate_adaptive_noise,
    apply_dp_noise_per_layer,
    quantize,
    generate_zero_sum_masks
)

DEVICE_MAP = {"raspberry_pi_4": 1, "raspberry_pi_zero": 2, "smartphone": 3, "iot_sensor": 4}

_profile_cache = {}

class QPrivIoTClient(NumPyClient):
    def __init__(self, context: Context):
        self.context = context
        self.run_config = context.run_config
        self.node_config = context.node_config

        self.partition_id = int(self.node_config.get("partition-id", 0))
        self.num_partitions = int(self.node_config.get("num-partitions", 1))

        if self.partition_id not in _profile_cache:
            _profile_cache[self.partition_id] = profile_device(partition_id=self.partition_id, config=CONFIG)
        self.profile = _profile_cache[self.partition_id]
        
        self.res_score = float(self.profile.get("resource_score", 0.0))
        self.device_type = self.profile.get("device_type", "unknown")

    def fit(self, parameters, config):
        """Train parameters on the locally held dataset."""
        # Default metric values for telemetry
        avg_sensitivity = 0.0
        avg_clip_norm = 0.0
        mean_noise = 0.0
        dp_mode = "None"
        dp_was_applied = False
        sensitivities_map = {}  # Initialize here so it's always in scope
        try:

            use_secagg = bool(config.get("use_secagg", False))
            use_dp = bool(config.get("use_dp", False))
            use_adaptive_dp = bool(config.get("use_adaptive_dp", False))

            base_latency = float(getattr(CONFIG.resource, "base_latency_seconds", 0.2)) / max(self.res_score, 0.01)
            time.sleep(min(base_latency, 0.5))

            cutoff = float(getattr(CONFIG.resource, "min_eligible_score", 0.2))
            if self.res_score < cutoff:
                print(
                    f"CLIENT {self.partition_id}: Skipped due to low resource score ({self.res_score:.2f} < {cutoff:.2f})")
                return parameters, 0, {
                    "resource_score": self.res_score,
                    "client_latency": base_latency,
                    "train_loss": 0.0,
                    "avg_noise": 0.0,
                    "secagg_active": False,
                    "dp_mode": "Skipped"
                }

            ds_name = str(self.run_config.get("dataset", "cifar10"))
            local_epochs = int(self.run_config.get("local-epochs", 1))
            # Use dynamic learning rate from server if available, otherwise fall back to run_config
            learning_rate = float(config.get("learning_rate", self.run_config.get("learning-rate", 0.01)))

            convergence_score = float(config.get("convergence_score", 0.0))
            fine_tuning_active = False
            if convergence_score > 0.8:
                learning_rate *= 0.5
                fine_tuning_active = True
                print(f"CLIENT {self.partition_id}: High convergence ({convergence_score:.2f}) - adjusting LR and noise for fine-tuning.")

            train_loader, _ = load_data(self.partition_id, self.num_partitions, 32, ds_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
            model = make_model(ds_name).to(device)

            self._set_parameters(model, parameters, device)

            initial_weights_list = [p.detach().clone() for _, p in model.named_parameters()]

            # SENSITIVITY METRICS (Now received from server or set to default)
            avg_sensitivity = 0.0
            noise_mults = {}
            clip_norms_map = {}
            dp_was_applied = False

            base_clip_norm_res = float(getattr(CONFIG.privacy, "initial_clip_norm", 0.05)) * max(self.res_score, 0.01)

            if use_adaptive_dp:
                # Extract AdaPriv parameters from config
                epsilon_t = float(config.get("epsilon_t", 1.0))
                sensitivities_str = config.get("sensitivities", "{}")
                try:
                    sensitivities_map = json.loads(sensitivities_str)
                except (json.JSONDecodeError, TypeError, ValueError):
                    sensitivities_map = {}

                # Adjust noise multiplier based on Readiness Score R_i
                # Section 3.5: Resource-constrained devices get higher noise to reduce their impact
                # Range [1.0, 2.0] chosen to balance privacy and utility:
                # - 1.0 for high-resource devices (normal noise)
                # - 2.0 for low-resource devices (double noise, less influence)
                # Note: This is an intentional design choice from Section 3.5.
                sigma_i_factor = 1.0 / (self.res_score + 1e-8)
                sigma_i_factor = np.clip(sigma_i_factor, 1.0, 2.0)
                
                # Allocate noise per layer based on sensitivities
                noise_mults, clip_norms_map, base_sigma = allocate_adaptive_noise(
                    sensitivities=sensitivities_map,
                    target_epsilon=epsilon_t,
                    base_clip_norm=base_clip_norm_res,
                )
                
                # If Round 1 (no sensitivities), use uniform base noise
                if not noise_mults:
                    param_keys = [n for n, _ in model.named_parameters()]
                    noise_mults = {k: base_sigma for k in param_keys}
                    clip_norms_map = {k: base_clip_norm_res for k in param_keys}

                # Scale noise multipliers by per-client factor sigma_i
                for k in noise_mults:
                    noise_mults[k] *= sigma_i_factor
                    
                if fine_tuning_active:
                    for k in noise_mults:
                        noise_mults[k] *= 0.8

                dp_was_applied = True
                dp_mode = "Adaptive"
                print(f"CLIENT {self.partition_id}: AdaPriv (ε_t={epsilon_t:.3f}, R_i={self.res_score:.2f})")

            elif use_dp:
                # Fixed DP: Calibrate noise to the per-round epsilon for a fair comparison
                epsilon_t = float(config.get("epsilon_t", 1.0))
                # base_sigma approx calculated using formal Gaussian mechanism
                fixed_noise_multiplier = math.sqrt(2 * math.log(1.25 / 1e-5)) / max(epsilon_t, 1e-6)
                fixed_clip_norm = 1.0
                
                param_keys = [n for n, _ in model.named_parameters()]
                noise_mults = {name: fixed_noise_multiplier for name in param_keys}
                if fine_tuning_active:
                    for k in noise_mults:
                        noise_mults[k] *= 0.8
                clip_norms_map = {name: fixed_clip_norm for name in param_keys}
                dp_was_applied = True
                dp_mode = "Fixed"
                print(f"CLIENT {self.partition_id}: Fixed DP (ε_t={epsilon_t:.3f}, Noise={fixed_noise_multiplier:.3f}, Clip={fixed_clip_norm:.2f})")

            else:
                dp_mode = "None"
                # Use a larger default clip norm for No DP to prevent explosion in FL
                # 5.0 chosen empirically for CIFAR-10 CNNs
                fixed_clip_norm = 5.0
                param_keys = [n for n, _ in model.named_parameters()]
                clip_norms_map = {name: fixed_clip_norm for name in param_keys}
                # Initialize empty noise multipliers for consistency
                noise_mults = {name: 0.0 for name in param_keys}
                print(f"CLIENT {self.partition_id}: Baseline Mode (No DP)")
            
            # Record performance metrics for telemetry
            avg_clip_norm_configured = float(np.mean(list(clip_norms_map.values()))) if clip_norms_map else 0.0

            # Training
            train_loss = float(train(model, train_loader, epochs=local_epochs, lr=learning_rate, device=device))

            observed_gradient_norms = []
            if dp_was_applied:
                num_samples = len(train_loader.dataset)
                # the function returns the modified weights, but we can intercept it to get norms if necessary
                # Instead, we will compute true observed norms just prior to calling apply_dp_noise
                with torch.no_grad():
                    # We ONLY apply DP to trainable parameters (named_parameters)
                    # so we should only track norms for those.
                    for idx_param, (_, param) in enumerate(model.named_parameters()):
                        if idx_param < len(initial_weights_list):
                            delta = param.data - initial_weights_list[idx_param]
                            observed_gradient_norms.append(torch.norm(delta).item())
                
                apply_dp_noise_per_layer(model, initial_weights_list, noise_mults, clip_norms_map, num_samples=num_samples)

            actual_avg_clip_norm = float(np.mean(observed_gradient_norms)) if observed_gradient_norms else avg_clip_norm_configured

            # Report mean noise for server-side privacy accounting
            mean_noise = float(np.mean(list(noise_mults.values()))) if noise_mults else 0.0

            # Compute deltas using named_parameters (trainable only) for consistency
            # with apply_dp_noise_per_layer which also uses named_parameters
            updated_trainable_params = [p.detach().cpu().numpy() for _, p in model.named_parameters()]
            initial_trainable_weights_numpy = [p.cpu().numpy() for p in initial_weights_list]

            updated_deltas = [
                updated_trainable_params[i] - initial_trainable_weights_numpy[i]
                for i in range(len(updated_trainable_params))
            ]

            if sensitivities_map:
                avg_sensitivity = float(np.mean(list(sensitivities_map.values())))

            metrics = {
                "train_loss": train_loss,
                "num_examples": len(train_loader.dataset),
                "resource_score": self.res_score, 
                "avg_noise": mean_noise,
                "client_latency": base_latency, 
                "secagg_active": use_secagg, 
                "dp_mode": dp_mode,
                "device_id": DEVICE_MAP.get(self.device_type, 0), 
                "avg_sensitivity": avg_sensitivity, 
                "clip_norm_configured": avg_clip_norm_configured,
                "clip_norm": actual_avg_clip_norm
            }

            if use_secagg:
                secagg_quantization_bound = float(config.get("secagg_quantization_bound", 10.0))
                metrics["secagg_quantization_bound"] = secagg_quantization_bound
                print(f"CLIENT {self.partition_id}: Applying SecAgg (Quantization bound: {secagg_quantization_bound})")

                secagg_seed = int(config.get("secagg_seed", 0))
                secagg_client_index = int(config.get("secagg_client_index", 0))
                secagg_total_clients = int(config.get("secagg_total_clients", 1))

                quantized_deltas = quantize(updated_deltas, clip_range=secagg_quantization_bound, range_max=1000000)

                shapes = [q.shape for q in quantized_deltas]

                all_masks = generate_zero_sum_masks(shapes, secagg_total_clients, secagg_seed)
                my_mask = all_masks[secagg_client_index]

                masked_parameters = []
                for i in range(len(quantized_deltas)):
                    masked_parameters.append(quantized_deltas[i].astype(np.int64) + my_mask[i].astype(np.int64))

                return masked_parameters, len(train_loader.dataset), metrics

            return updated_deltas, len(train_loader.dataset), metrics

        except Exception as e:
            print(f"CLIENT EXCEPTION in partition {self.partition_id}: {e}")
            traceback.print_exc()
            raise e

    def evaluate(self, parameters, config):
        try:
            ds_name = str(self.run_config.get("dataset", "cifar10"))
            _, val_loader = load_data(self.partition_id, self.num_partitions, 32, ds_name)

            device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
            model = make_model(ds_name).to(device)
            self._set_parameters(model, parameters, device)

            loss, accuracy = test(model, val_loader, device)

            return float(loss), len(val_loader.dataset), {"accuracy": float(accuracy)}
        except Exception as e:
            print(f"CLIENT EVAL EXCEPTION: {e}")
            raise e

    def _set_parameters(self, model, parameters, device):
        """Sets model parameters from a list of NumPy arrays."""
        set_weights(model, parameters)

    def _get_parameters(self, model):
        """Returns model parameters as a list of NumPy arrays."""
        return get_weights(model)


def client_fn(context: Context):
    return QPrivIoTClient(context).to_client()


app = ClientApp(client_fn=client_fn)
