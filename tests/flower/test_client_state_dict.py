import numpy as np, torch
from qpriviot_fl.task import make_model
from qpriviot_fl.client_app import QPrivIoTClient

class _Ctx:
    run_config = {"dataset": "cifar10", "local-epochs": 1, "learning-rate": 0.01}
    node_config = {"partition-id": 0, "num-partitions": 1}

class BNNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 3)
        self.bn = torch.nn.BatchNorm2d(6)
    def forward(self, x):
        return self.bn(self.conv(x))

def test_parameter_roundtrip_preserves_buffers(monkeypatch):
    model_a = BNNet()
    # Set running_mean to a non-zero value to test buffer preservation
    model_a.bn.running_mean.fill_(1.23)
    
    # We use state_dict now to include buffers
    arrays = [val.cpu().numpy() for _, val in model_a.state_dict().items()]
    model_b = BNNet()
    client = QPrivIoTClient.__new__(QPrivIoTClient)  # bypass __init__
    client._set_parameters(model_b, arrays, torch.device("cpu"))
    for (n, a), (_, b) in zip(model_a.state_dict().items(), model_b.state_dict().items()):
        np.testing.assert_allclose(a.cpu().numpy(), b.cpu().numpy())
    
    assert model_b.bn.running_mean[0] == 1.23
