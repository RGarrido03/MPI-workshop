import torch
import torch.nn as nn
from torch import device as Device

from data import get_dataloader


def test_model(model: nn.Module, device: Device) -> float:
    dataloader = get_dataloader("test")
    correct, total = 0, 0

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy}%")
    return accuracy
