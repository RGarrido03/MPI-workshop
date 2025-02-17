import time

import torch.nn as nn
from torch import device as Device
from torch import optim

from data import get_dataloader


def train_model(
    model: nn.Module,
    num_epochs: int,
    device: Device,
):
    dataloader = get_dataloader("train")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        end_time = time.time()
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}, Time: {end_time-start_time:.2f}s"
        )
