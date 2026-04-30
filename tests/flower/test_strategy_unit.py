import numpy as np, pytest
from unittest.mock import MagicMock
from flwr.common import ndarrays_to_parameters, FitRes, Status, Code
from qpriviot_fl.server_app import ProgressivePrivacyStrategy
from qpriviot_fl.task import make_model, get_weights

def _mock_fitres(deltas, n=100, **metrics):
    return FitRes(status=Status(Code.OK, ""), parameters=ndarrays_to_parameters(deltas),
                  num_examples=n, metrics={"train_loss": 0.5, "resource_score": 0.8,
                                           "avg_noise": 1.0, "dp_mode": "Fixed",
                                           **metrics})

def test_aggregate_returns_none_on_empty():
    s = ProgressivePrivacyStrategy(num_rounds=1, dataset="cifar10")
    assert s.aggregate_fit(1, [], []) == (None, {})

def test_weighted_avg_two_clients():
    s = ProgressivePrivacyStrategy(num_rounds=1, dataset="cifar10",
                                   use_adaptive_dp=False, target_epsilon=100.0)
    model = make_model("cifar10")
    init_params = get_weights(model)
    s.current_parameters = ndarrays_to_parameters(init_params)
    d1 = [np.ones_like(p) for p in init_params]
    d2 = [np.full_like(p, 3.0) for p in init_params]
    p1, p2 = MagicMock(cid="c1"), MagicMock(cid="c2")
    res1, res2 = _mock_fitres(d1, n=100), _mock_fitres(d2, n=300)
    out, _ = s.aggregate_fit(1, [(p1, res1), (p2, res2)], [])
    # Weighted by num_examples: (1*100 + 3*300) / 400 = 2.5
    from flwr.common import parameters_to_ndarrays
    out_arrays = list(parameters_to_ndarrays(out))
    np.testing.assert_allclose(out_arrays[0], init_params[0] + 2.5)
