import torch.multiprocessing as mp
import os
from src.distributed.comm import init_dist, cleanup_dist, all_reduce_sum
import torch

def worker_fn(rank, world_size, logic_fn):
    """Wrapper que inicializa o processo e executa a lógica do worker."""
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    init_dist(rank, world_size)
    
    # Executa a lógica passada
    logic_fn(rank, world_size)
    
    cleanup_dist()

class DistCoordinator:
    def __init__(self, world_size=2):
        self.world_size = world_size

    def run(self, logic_fn):
        mp.spawn(
            worker_fn,
            args=(self.world_size, logic_fn),
            nprocs=self.world_size,
            join=True
        )
