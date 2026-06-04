import math
import time
import torch
import numpy as np
import traceback
from datetime import datetime
from flwr.client import NumPyClient
from flwr.clientapp import ClientApp
from flwr.common import Context
import json

from qpriviot_fl.config import DEFAULT_CONFIG as CONFIG
from qpriviot_fl.device_profile import profile_device
from qpriviot_fl.task import load_data, make_model, train, test
from qpriviot_fl.privacy_utils import (
    allocate_adaptive_noise,
    apply_dp_noise_per_layer,
    quantize,
    generate_zero_sum_masks
)

DEVICE_MAP = {"raspberry_pi_4": 1, "raspberry_pi_zero": 2, "smartphone": 3, "iot_sensor": 4}


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


class QPrivIoTClient(NumPyClient):
    def __init__(self, context: Context):
        self.context = context
        self.run_config = context.run_config
        self.node_config = context.node_config

        self.partition_id = int(self.node_config.get("partition-id", 0))
        self.num_partitions = int(self.node_config.get("num-partitions", 1))

        self.profile = profile_device(partition_id=self.partition_id, config=CONFIG)
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
                _log(f"CLIENT {self.partition_id}: Skipped due to low resource score ({self.res_score:.2f} < {cutoff:.2f})")
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

            dirichlet_alpha = float(config.get("dirichlet_alpha", 0.3))
            seed = int(config.get("seed", 42))

            convergence_score = float(config.get("convergence_score", 0.0))
            fine_tuning_active = False
            if convergence_score > 0.8:
                learning_rate *= 0.5
                fine_tuning_active = True
                _log(f"CLIENT {self.partition_id}: High convergence ({convergence_score:.2f}) - adjusting LR and noise for fine-tuning.")

            train_loader, val_loader = load_data(self.partition_id, self.num_partitions, 32, ds_name,
                                               alpha=dirichlet_alpha, seed=seed)
            device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
            model = make_model(ds_name).to(device)

            self._set_parameters(model, parameters, device)

            initial_weights_list = [p.detach().clone() for _, p in model.named_parameters()]

            # SENSITIVITY METRICS (Now received from server or set to default)
            avg_sensitivity = 0.0
            noise_mults = {}
            clip_norms_map = {}
            dp_was_applied = False

            # Clip norm must be independent of resource score — scaling it down by R_i
            # (which is 0.1–0.4 for simulated IoT) would make clip_norm ~0.005 and completely
            # bury the gradient under Opacus noise (σ≈6–10). Resource-tier effect belongs
            # only in sigma_i_factor (noise), not clipping.
            base_clip_norm_res = float(getattr(CONFIG.privacy, "initial_clip_norm", 1.0))

            epsilon_t = float(config.get("epsilon_t", 1.0))
            ablation_mode = str(config.get("ablation_mode", ""))
            if use_adaptive_dp:
                # Extract AdaPriv parameters from config
                sensitivities_str = config.get("sensitivities", "{}")
                try:
                    sensitivities_map = json.loads(sensitivities_str)
                except (json.JSONDecodeError, TypeError, ValueError):
                    sensitivities_map = {}

                opacus_sigma = float(config.get("sigma", 0.0))
                param_keys = [n for n, _ in model.named_parameters()]

                if ablation_mode == "device-only":
                    # Device-tier scaling only — uniform layers, no per-layer / round adaptation.
                    # Use relative formula avg/res_i so that above-average devices get factor≈1.0
                    # and below-average devices get proportionally more noise. The absolute 1/res_i
                    # formula saturates at the cap for every simulated IoT device (verified: all
                    # 10 partitions get exactly 1.3×), eliminating any device differentiation.
                    _, _, base_sigma = allocate_adaptive_noise(
                        sensitivities={}, target_epsilon=epsilon_t,
                        base_clip_norm=base_clip_norm_res,
                        base_sigma=opacus_sigma if opacus_sigma > 0 else None,
                    )
                    avg_resource_score = float(config.get("avg_resource_score", 0.3))
                    # Fairness direction (flipped): low-resource clients get LESS noise
                    # so their gradient SNR improves and their accuracy catches up to
                    # well-resourced peers. The ratio res_i/avg has cohort-mean ≈ 1.0,
                    # so the *average* noise budget is conserved; weak clients dip below
                    # base, strong clients rise above it.
                    sigma_i_factor = float(np.clip(self.res_score / (avg_resource_score + 1e-8), 0.7, 1.3))
                    noise_mults = {k: base_sigma * sigma_i_factor for k in param_keys}
                    clip_norms_map = {k: base_clip_norm_res for k in param_keys}

                elif ablation_mode == "param-only":
                    # Per-layer sensitivity only — no device scaling, no round schedule.
                    # Biases are excluded from sensitivity-based allocation: they are global
                    # offsets that don't interact with individual input features and have
                    # negligible per-sample privacy leakage. Adaptive allocation applies
                    # only to weight matrices; biases receive base (fixed-DP equivalent) noise.
                    weight_sensitivities = {k: v for k, v in sensitivities_map.items()
                                            if not k.endswith('.bias')}
                    noise_mults, clip_norms_map, base_sigma = allocate_adaptive_noise(
                        sensitivities=weight_sensitivities, target_epsilon=epsilon_t,
                        base_clip_norm=base_clip_norm_res,
                        base_sigma=opacus_sigma if opacus_sigma > 0 else None,
                    )
                    for k in param_keys:
                        if k not in noise_mults:
                            noise_mults[k] = base_sigma
                            clip_norms_map[k] = base_clip_norm_res
                    # no sigma_i_factor applied

                elif ablation_mode == "round-only":
                    # Round schedule only — no per-layer or device adaptation
                    _, _, base_sigma = allocate_adaptive_noise(
                        sensitivities={}, target_epsilon=epsilon_t,
                        base_clip_norm=base_clip_norm_res,
                        base_sigma=opacus_sigma if opacus_sigma > 0 else None,
                    )
                    noise_mults = {k: base_sigma for k in param_keys}
                    clip_norms_map = {k: base_clip_norm_res for k in param_keys}
                    # Apply round noise factor: epsilon_t from cosine schedule drives noise up/down
                    _num_rounds = int(config.get("num_rounds", self.run_config.get("num-server-rounds", 100)))
                    _base_eps = float(config.get("target_epsilon", 3.0)) / max(_num_rounds, 1)
                    _round_noise_factor = float(np.clip(_base_eps / max(epsilon_t, 1e-6), 0.5, 2.0))
                    for k in noise_mults:
                        noise_mults[k] *= _round_noise_factor

                else:
                    # Full AdaPriv: per-layer + device + round.
                    # Same bias-exclusion rationale as param-only: biases receive base noise.
                    weight_sensitivities = {k: v for k, v in sensitivities_map.items()
                                            if not k.endswith('.bias')}
                    noise_mults, clip_norms_map, base_sigma = allocate_adaptive_noise(
                        sensitivities=weight_sensitivities, target_epsilon=epsilon_t,
                        base_clip_norm=base_clip_norm_res,
                        base_sigma=opacus_sigma if opacus_sigma > 0 else None,
                    )
                    for k in param_keys:
                        if k not in noise_mults:
                            noise_mults[k] = base_sigma
                            clip_norms_map[k] = base_clip_norm_res
                    # Device-tier: normalize against the cohort average so high-resource clients
                    # get factor≈1.0 and low-resource clients get proportionally more noise.
                    # Uses relative scaling (avg/res_i) instead of absolute 1/res_i to avoid
                    # constant saturation when every client has a low absolute res_score.
                    avg_resource_score = float(config.get("avg_resource_score", 0.3))
                    # Fairness direction (flipped — see device-only branch): low-resource
                    # clients get LESS noise (factor<1), high-resource clients absorb MORE.
                    # Cohort-mean of res_i/avg ≈ 1.0 keeps the average noise budget conserved.
                    sigma_i_factor = float(np.clip(self.res_score / (avg_resource_score + 1e-8), 0.7, 1.3))
                    for k in noise_mults:
                        noise_mults[k] *= sigma_i_factor
                    # Round schedule: cosine-annealed epsilon_t drives noise — more noise when
                    # budget is tight (late rounds), less when budget is ample (early rounds).
                    _num_rounds = int(config.get("num_rounds", self.run_config.get("num-server-rounds", 100)))
                    _base_eps = float(config.get("target_epsilon", 3.0)) / max(_num_rounds, 1)
                    _round_noise_factor = float(np.clip(_base_eps / max(epsilon_t, 1e-6), 0.5, 2.0))
                    for k in noise_mults:
                        noise_mults[k] *= _round_noise_factor

                if fine_tuning_active and not ablation_mode:
                    for k in noise_mults:
                        noise_mults[k] *= 0.8

                dp_was_applied = True
                dp_mode = "Adaptive" if not ablation_mode else f"Ablation-{ablation_mode}"
                _log(f"CLIENT {self.partition_id}: AdaPriv mode={ablation_mode or 'full'} (ε_t={epsilon_t:.3f}, R_i={self.res_score:.2f})")

            elif use_dp:
                # Fixed DP: Use the pre-calibrated sigma from server if available
                sigma = float(config.get("sigma", 0.0))
                if sigma == 0:
                    epsilon_t = float(config.get("epsilon_t", 1.0))
                    sigma = math.sqrt(2 * math.log(1.25 / 1e-5)) / max(epsilon_t, 1e-6)
                
                fixed_noise_multiplier = sigma
                fixed_clip_norm = 1.0
                
                param_keys = [n for n, _ in model.named_parameters()]
                noise_mults = {name: fixed_noise_multiplier for name in param_keys}
                if fine_tuning_active:
                    for k in noise_mults:
                        noise_mults[k] *= 0.8
                clip_norms_map = {name: fixed_clip_norm for name in param_keys}
                dp_was_applied = True
                dp_mode = "Fixed"
                _log(f"CLIENT {self.partition_id}: Fixed DP (Noise={fixed_noise_multiplier:.3f}, Clip={fixed_clip_norm:.2f})")

            else:
                dp_mode = "None"
                # Use a larger default clip norm for No DP to prevent explosion in FL
                # 5.0 chosen empirically for CIFAR-10 CNNs
                fixed_clip_norm = 5.0
                param_keys = [n for n, _ in model.named_parameters()]
                clip_norms_map = {name: fixed_clip_norm for name in param_keys}
                # Initialize empty noise multipliers for consistency
                noise_mults = {name: 0.0 for name in param_keys}
                _log(f"CLIENT {self.partition_id}: Baseline Mode (No DP)")
            
            # Record performance metrics for telemetry
            avg_clip_norm_configured = float(np.mean(list(clip_norms_map.values()))) if clip_norms_map else 0.0

            # Evaluate on validation set BEFORE applying DP noise.
            # Reuse the existing model + pre-loaded val_loader to avoid a second model instantiation.
            pre_val_loss, pre_val_acc = test(model, val_loader, device)
            val_loss = pre_val_loss
            val_num_samples = len(val_loader.dataset)
            eval_metrics_dict = {"accuracy": pre_val_acc}

            mu = float(config.get("proximal_mu", 0.0))

            # Training
            train_loss = float(train(model, train_loader, epochs=local_epochs, lr=learning_rate, device=device,
                                     global_state=initial_weights_list, mu=mu))

            observed_gradient_norms = []
            if dp_was_applied:
                num_samples = len(train_loader.dataset)
                # the function returns the modified weights, but we can intercept it to get norms if necessary
                # Instead, we will compute true observed norms just prior to calling apply_dp_noise
                with torch.no_grad():
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
            updated_params = [p.detach().cpu().numpy() for _, p in model.named_parameters()]
            initial_weights_numpy = [p.cpu().numpy() for p in initial_weights_list]

            updated_deltas = [
                updated_params[i] - initial_weights_numpy[i]
                for i in range(len(updated_params))
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
                "val_accuracy": float(eval_metrics_dict.get("accuracy", 0.0)),
                "val_loss": float(val_loss), 
                "avg_sensitivity": avg_sensitivity, 
                "clip_norm_configured": avg_clip_norm_configured,
                "clip_norm": actual_avg_clip_norm
            }

            if use_secagg:
                secagg_quantization_bound = float(config.get("secagg_quantization_bound", 10.0))
                metrics["secagg_quantization_bound"] = secagg_quantization_bound
                _log(f"CLIENT {self.partition_id}: Applying SecAgg (Quantization bound: {secagg_quantization_bound})")

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

                del model
                return masked_parameters, len(train_loader.dataset), metrics

            del model
            return updated_deltas, len(train_loader.dataset), metrics

        except Exception as e:
            _log(f"CLIENT EXCEPTION in partition {self.partition_id}: {e}")
            traceback.print_exc()
            raise e

    def evaluate(self, parameters, config):
        try:
            ds_name = str(self.run_config.get("dataset", "cifar10"))
            dirichlet_alpha = float(config.get("dirichlet_alpha", 0.3))
            seed = int(config.get("seed", 42))
            
            _, val_loader = load_data(self.partition_id, self.num_partitions, 32, ds_name,
                                     alpha=dirichlet_alpha, seed=seed)

            device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
            model = make_model(ds_name).to(device)
            self._set_parameters(model, parameters, device)

            loss, accuracy = test(model, val_loader, device)

            return float(loss), len(val_loader.dataset), {"accuracy": float(accuracy)}
        except Exception as e:
            _log(f"CLIENT EVAL EXCEPTION: {e}")
            raise e

    def _set_parameters(self, model, parameters, device):
        """Sets model parameters from a list of NumPy arrays."""
        for (_, param), new_val in zip(model.named_parameters(), parameters):
            tensor = torch.from_numpy(new_val).to(device=device, dtype=param.dtype)
            param.data = tensor.view(param.shape)

    def _get_parameters(self, model):
        """Returns model parameters as a list of NumPy arrays."""
        return [val.detach().cpu().numpy() for _, val in model.named_parameters()]


def client_fn(context: Context):
    return QPrivIoTClient(context).to_client()


app = ClientApp(client_fn=client_fn)
