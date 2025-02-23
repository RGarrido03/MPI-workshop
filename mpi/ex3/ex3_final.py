import numpy as np
from mpi4py import MPI


def monte_carlo_pi_simulation(num_samples: int) -> int:
    count = 0
    for _ in range(num_samples):
        x, y = np.random.uniform(-1, 1, 2)
        if x**2 + y**2 <= 1:
            count += 1
    return count


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 10**6
rank_samples = N // size
rank_count = monte_carlo_pi_simulation(rank_samples)

total_count = comm.reduce(rank_count, op=MPI.SUM, root=0)

if rank == 0:
    pi_estimate = (4 * total_count) / N
    print(f"Estimated pi value: {pi_estimate}")
