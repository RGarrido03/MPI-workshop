from mpi4py import MPI

world_comm = MPI.COMM_WORLD
my_rank = world_comm.Get_rank()
