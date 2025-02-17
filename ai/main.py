# import torch.distributed as dist
import torch
from mpi4py import MPI

from model import CNNModel
from testing import test_model
from training import train_model

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# dist.init_process_group(backend="mpi")

model = CNNModel()
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
model.to(device)

"""
Train the model and save it, or load it from a file.
(Un)comment the appropriate lines.
"""
train_model(model, 70, device)
torch.save(model.state_dict(), "cnn_model.pth")
# model.load_state_dict(torch.load("cnn_model.pth"))


test_model(model, device)
