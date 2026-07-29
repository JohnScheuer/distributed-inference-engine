import torch
import torch.distributed as dist
from src.distributed.config import DistConfig

def init_dist(rank: int, world_size: int):
    config = DistConfig(rank=rank, world_size=world_size)
    dist.init_process_group(
        backend=config.backend,
        init_method=f"tcp://{config.master_addr}:{config.master_port}",
        rank=rank,
        world_size=world_size
    )
    print(f"[Rank {rank}] Distributed initialized.")

def all_reduce_sum(tensor: torch.Tensor):
    """Soma o tensor em todos os ranks e sincroniza o resultado."""
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor

def cleanup_dist():
    dist.destroy_process_group()
