import numpy as np
from mpi4py import MPI


def compute_range(rank: int, size: int, total_work: int) -> tuple[int, int, int]:
    """
    Compute the start (inclusive) and end (exclusive) indices of the workload for a given rank.
    """
    base, extra = divmod(total_work, size)
    start = rank * base + min(rank, extra)
    end = start + base + (1 if rank < extra else 0)
    return start, end, end - start


def initialize_array(size: int, start: int) -> np.ndarray:
    """
    Initialize an array of values, with a defined size.
    The array is populated with values starting from `start`.
    """
    arr = np.ones(size)
    for i in range(size):
        arr[i] += i + start
    return arr


world_comm = MPI.COMM_WORLD
world_size = world_comm.Get_size()
my_rank = world_comm.Get_rank()
N = 1000000

my_start, my_end, size = compute_range(my_rank, world_size, N)
a = initialize_array(size, my_start)

rank_sum = np.array([sum(a)])
world_sum = np.zeros(1)
world_comm.Reduce([rank_sum, MPI.DOUBLE], [world_sum, MPI.DOUBLE], op=MPI.SUM, root=0)

if my_rank == 0:
    average = world_sum[0] / N
    print(f"Average: {average}")
