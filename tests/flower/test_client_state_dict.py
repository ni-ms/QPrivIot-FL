import numpy as np, torch
from qpriviot_fl.task import make_model
from qpriviot_fl.client_app import QPrivIoTClient

class _Ctx:
    run_config = {"dataset": "cifar10", "local-epochs": 1, "learning-rate": 0.01}
    node_config = {"partition-id": 0, "num-partitions": 1}

def test_parameter_roundtrip_preserves_buffers(monkeypatch):
    model_a = make_model("cifar10")
    # We use state_dict now to include buffers
    arrays = [val.cpu().numpy() for _, val in model_a.state_dict().items()]
    model_b = make_model("cifar10")
    client = QPrivIoTClient.__new__(QPrivIoTClient)  # bypass __init__
    client._set_parameters(model_b, arrays, torch.device("cpu"))
    for (n, a), (_, b) in zip(model_a.state_dict().items(), model_b.state_dict().items()):
        np.testing.assert_allclose(a.detach().numpy(), b.detach().numpy())
