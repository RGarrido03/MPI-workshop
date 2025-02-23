from mpi4py import MPI

world_comm = MPI.COMM_WORLD
my_rank = world_comm.Get_rank()

match my_rank:
    case 0:
        data = {"pi": 3.1416, "e": 2.7183}
        world_comm.send(data, dest=1)
        print(f"Rank {my_rank} sent data: {str(data)}")
    case 1:
        data = world_comm.recv()
        print(f"Rank {my_rank} received data: {str(data)}")
    case _:
        print(f"Rank {my_rank} did nothing")
