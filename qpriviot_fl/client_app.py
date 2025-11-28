import math
import time
import torch
import numpy as np
import traceback
from flwr.client import NumPyClient
from flwr.clientapp import ClientApp
from flwr.common import Context

from qpriviot_fl.config import DEFAULT_CONFIG as CONFIG
from qpriviot_fl.device_profile import profile_device
from qpriviot_fl.task import load_data, make_model, train, test
from qpriviot_fl.privacy_utils import (
    compute_gradient_sensitivity,
    allocate_adaptive_noise,
    apply_dp_noise_per_layer,
    quantize,
    generate_zero_sum_masks
)

DEVICE_MAP = {"raspberry_pi_4": 1, "raspberry_pi_zero": 2, "smartphone": 3, "iot_sensor": 4}


class QPrivIoTClient(NumPyClient):
    def __init__(self, context: Context):
        self.context = context
        self.run_config = context.run_config
        self.node_config = context.node_config

        self.profile = profile_device(CONFIG)
        self.res_score = float(self.profile.get("resource_score", 0.0))
        self.device_type = self.profile.get("device_type", "unknown")
        self.partition_id = int(self.node_config.get("partition-id", 0))
        self.num_partitions = int(self.node_config.get("num-partitions", 1))

    def fit(self, parameters, config):
        """Train parameters on the locally held dataset."""
        try:

            use_secagg = bool(config.get("use_secagg", False))
            use_dp = bool(config.get("use_dp", False))
            use_adaptive_dp = bool(config.get("use_adaptive_dp", False))

            base_noise = float(self.run_config.get("base-noise", 1.2))

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
            learning_rate = float(self.run_config.get("learning-rate", 0.01))

            train_loader, _ = load_data(self.partition_id, self.num_partitions, 32, ds_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "mps")
            model = make_model(ds_name).to(device)

            self._set_parameters(model, parameters, device)

            initial_weights_list = [p.detach().clone() for p in model.parameters()]

            layer_sensitivities = compute_gradient_sensitivity(model, train_loader, device=device)

            # CAPTURE SENSITIVITY METRICS
            avg_sensitivity = float(np.mean(list(layer_sensitivities.values()))) if layer_sensitivities else 0.0


            noise_mults = {}
            clip_norms_map = {}
            dp_was_applied = False

            base_clip_norm_res = float(getattr(CONFIG.privacy, "initial_clip_norm", 0.05)) * max(self.res_score, 0.01)

            if use_adaptive_dp:

                server_convergence = float(config.get("convergence_score", 0.0))
                target_eps = float(self.run_config.get("target-epsilon", 10.0))
                total_rounds = int(self.run_config.get("num-server-rounds", 10))

                round_base_eps = target_eps / math.sqrt(max(total_rounds, 1))
                adaptive_round_eps = round_base_eps * (1.0 + server_convergence)

                noise_mults, clip_norms_map, _ = allocate_adaptive_noise(
                    sensitivities=layer_sensitivities,
                    target_epsilon=adaptive_round_eps,
                    base_clip_norm=base_clip_norm_res,
                )
                dp_was_applied = True
                dp_mode = "Adaptive"
                print(f"CLIENT {self.partition_id}: Adaptive DP (Conv={server_convergence:.2f})")

            elif use_dp:

                noise_mults = {name: base_noise for name in layer_sensitivities.keys()}

                fixed_clip_norm = base_clip_norm_res * 2
                clip_norms_map = {name: fixed_clip_norm for name in layer_sensitivities.keys()}
                dp_was_applied = True
                dp_mode = "Fixed"
                print(f"CLIENT {self.partition_id}: Fixed DP (Noise={base_noise:.3f})")

            else:

                fixed_clip_norm = 10.0
                clip_norms_map = {name: fixed_clip_norm for name in layer_sensitivities.keys()}
                dp_mode = "None"
                print(f"CLIENT {self.partition_id}: Baseline Mode (No DP)")
                # CAPTURE CLIPPING NORMS
            avg_clip_norm = float(np.mean(list(clip_norms_map.values()))) if clip_norms_map else 0.0

            train_loss = float(train(model, train_loader, epochs=local_epochs, lr=learning_rate, device=device))

            if dp_was_applied:
                apply_dp_noise_per_layer(model, initial_weights_list, noise_mults, clip_norms_map)

                # EVALUATE ON VALIDATION SET BEFORE RETURNING
            eval_params = self._get_parameters(model)
            val_loss, val_num_samples, eval_metrics_dict = self.evaluate(eval_params, config)

            updated_params_final = self._get_parameters(model)
            initial_weights_numpy = [p.cpu().numpy() for p in initial_weights_list]

            updated_deltas = [
                updated_params_final[i] - initial_weights_numpy[i]
                for i in range(len(updated_params_final))
            ]

            avg_noise = float(np.mean(list(noise_mults.values()))) if noise_mults else 0.0

            metrics = {"train_loss": train_loss, "resource_score": self.res_score, "avg_noise": avg_noise,
                       "client_latency": base_latency, "secagg_active": use_secagg, "dp_mode": dp_mode,
                       "device_id": DEVICE_MAP.get(self.device_type, 0), "evaluation_metrics": eval_metrics_dict,
                       "val_loss": float(val_loss), "avg_sensitivity": avg_sensitivity, "clip_norm": avg_clip_norm}

            if use_secagg:
                print(f"CLIENT {self.partition_id}: Applying SecAgg (Quantization and Masking)")

                secagg_seed = int(config.get("secagg_seed", 0))
                secagg_client_index = int(config.get("secagg_client_index", 0))
                secagg_total_clients = int(config.get("secagg_total_clients", 1))

                quantized_deltas = quantize(updated_deltas, clip_range=1.0, range_max=1000000)

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

            device = torch.device("cuda" if torch.cuda.is_available() else "mps")
            model = make_model(ds_name).to(device)
            self._set_parameters(model, parameters, device)

            loss, accuracy = test(model, val_loader, device)

            return float(loss), len(val_loader.dataset), {"accuracy": float(accuracy)}
        except Exception as e:
            print(f"CLIENT EVAL EXCEPTION: {e}")
            raise e

    def _set_parameters(self, model, parameters, device):
        """Sets model parameters from a list of NumPy arrays."""
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {}
        for k, v in params_dict:
            tensor = torch.from_numpy(v).to(device=device, dtype=model.state_dict()[k].dtype)
            state_dict[k] = tensor.view(model.state_dict()[k].shape)
        model.load_state_dict(state_dict, strict=True)

    def _get_parameters(self, model):
        """Returns model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in model.state_dict().items()]


def client_fn(context: Context):
    return QPrivIoTClient(context).to_client()


app = ClientApp(client_fn=client_fn)
