import random
import numpy as np
import torch
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: integration tests (>30s)")

@pytest.fixture(autouse=True)
def _deterministic():
    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1337)
