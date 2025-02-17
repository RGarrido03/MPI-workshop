# import torch.distributed as dist
import time

import torch
import torch.nn as nn
from mpi4py import MPI
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

from constants import IMAGE_SIZE
from data import MyDataset
from model import CNNModel

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# dist.init_process_group(backend="mpi")

model = CNNModel()


device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda")
)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)

transform = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor()])
dataset = MyDataset("train", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


num_epochs = 70
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

print("Finished training, saving model")
torch.save(model.state_dict(), "cnn_model.pth")
