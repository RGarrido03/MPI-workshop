import torch
# import torch.distributed as dist
from mpi4py import MPI
from torch.utils.data import DataLoader
from torchvision import transforms

from constants import IMAGE_SIZE
from data import MyDataset
from model import CNNModel
from training import train_model

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# dist.init_process_group(backend="mpi")

model = CNNModel()


device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda")
)
model.to(device)
transform = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor()])


"""
Train the model and save it, or load it from a file.
(Un)comment the appropriate lines.
"""
dataset = MyDataset("train", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
train_model(model, dataloader, 70, device)
torch.save(model.state_dict(), "cnn_model.pth")
# model.load_state_dict(torch.load("cnn_model.pth"))
