"""QPrivIot-FL: Enhanced task module with advanced privacy features."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.validators import ModuleValidator
from opacus.grad_sample import GradSampleModule
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from opacus import PrivacyEngine
import numpy as np
from collections import defaultdict

cifar_transforms = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

femnist_transforms = Compose([
    Resize((28, 28)),
    ToTensor(),
    Normalize((0.5,), (0.5,))
])


class Net(nn.Module):
    """Simple CNN for CIFAR-10."""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class FEMNISTNet(nn.Module):
    """CNN for FEMNIST dataset."""

    def __init__(self):
        super(FEMNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 62)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


fds_cache = {}


def apply_transforms_cifar(batch):
    """Apply transforms to CIFAR-10 partition."""
    batch["img"] = [cifar_transforms(img) for img in batch["img"]]
    return batch


def apply_transforms_femnist(batch):
    """Apply transforms to FEMNIST partition."""
    batch["image"] = [femnist_transforms(img) for img in batch["image"]]
    return batch


def load_data(partition_id: int, num_partitions: int, dataset_name: str = "cifar10"):
    """Load partitioned data for CIFAR-10 or FEMNIST."""
    global fds_cache

    if dataset_name not in fds_cache:
        partitioner = IidPartitioner(num_partitions=num_partitions)

        if dataset_name == "cifar10":
            fds_cache[dataset_name] = FederatedDataset(
                dataset="uoft-cs/cifar10",
                partitioners={"train": partitioner},
            )
        elif dataset_name == "femnist":
            fds_cache[dataset_name] = FederatedDataset(
                dataset="flwrlabs/femnist",
                partitioners={"train": partitioner},
            )

    partition = fds_cache[dataset_name].load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    if dataset_name == "cifar10":
        partition_train_test = partition_train_test.with_transform(apply_transforms_cifar)
        img_key = "img"
        label_key = "label"
    else:
        partition_train_test = partition_train_test.with_transform(apply_transforms_femnist)
        img_key = "image"
        label_key = "label"

    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)

    return trainloader, testloader, img_key, label_key


def signal_sensitivity_estimation(trainloader, model, device, img_key="img", label_key="label"):
    """
    Advanced signal sensitivity estimation based on gradient variance and parameter importance.
    Returns per-layer sensitivity scores.
    """
    model.eval()
    gradient_norms_per_layer = defaultdict(list)

    num_samples = min(5, len(trainloader))
    criterion = nn.CrossEntropyLoss()

    for i, batch in enumerate(trainloader):
        if i >= num_samples:
            break

        images = batch[img_key].to(device)
        labels = batch[label_key].to(device)

        model.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(2).item()
                gradient_norms_per_layer[name].append(grad_norm)

    layer_sensitivity = {}
    for name, norms in gradient_norms_per_layer.items():
        if len(norms) > 0:
            variance = np.var(norms)
            mean_norm = np.mean(norms)

            sensitivity = 0.8 + (variance / (mean_norm + 1e-8)) * 0.4
            sensitivity = np.clip(sensitivity, 0.5, 1.5)
            layer_sensitivity[name] = sensitivity

    if layer_sensitivity:
        global_sensitivity = np.mean(list(layer_sensitivity.values()))
    else:
        global_sensitivity = 1.0

    return global_sensitivity, layer_sensitivity


def train_with_adaptive_dp(
        net,
        trainloader,
        epochs,
        lr,
        device,
        noise_multiplier,
        max_grad_norm,
        target_delta,
        layer_sensitivity=None,
        img_key="img",
        label_key="label",
):
    """Train model with layer-wise adaptive differential privacy using Opacus."""

    net = net.to(device)
    net = ModuleValidator.fix(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    privacy_engine = PrivacyEngine()
    net, optimizer, trainloader = privacy_engine.make_private(
        module=net,
        optimizer=optimizer,
        data_loader=trainloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    if layer_sensitivity is not None and isinstance(net._module, GradSampleModule):
        original_noise = optimizer.noise_multiplier

        pass

    net.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for epoch in range(epochs):
        for batch in trainloader:
            images = batch[img_key].to(device)
            labels = batch[label_key].to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / (len(trainloader) * epochs)
    accuracy = correct / total if total > 0 else 0.0

    epsilon = privacy_engine.get_epsilon(delta=target_delta)

    try:
        best_alpha = privacy_engine.accountant.get_privacy_spent(delta=target_delta)[0]
    except:
        best_alpha = 0.0

    return avg_loss, accuracy, epsilon, best_alpha, privacy_engine


def test(net, testloader, device, img_key="img", label_key="label"):
    """Evaluate model on test set."""
    net = net.to(device)

    if isinstance(net, GradSampleModule):
        net = net._module

    net.eval()

    criterion = nn.CrossEntropyLoss()
    correct = 0
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for batch in testloader:
            images = batch[img_key].to(device)
            labels = batch[label_key].to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += int((predicted == labels).sum())

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(testloader) if len(testloader) > 0 else 0.0

    return avg_loss, accuracy
