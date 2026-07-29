import torch
from src.distributed.coordinator import DistCoordinator
from src.distributed.comm import all_reduce_sum

def simple_sum_logic(rank, world_size):
    # Cada rank cria um tensor com o seu próprio rank
    data = torch.tensor([float(rank)], device="cpu")
    print(f"Rank {rank} data before: {data.item()}")
    
    # AllReduce: soma os valores de todos os ranks
    # Rank 0 (0.0) + Rank 1 (1.0) = 1.0 em ambos os processos
    all_reduce_sum(data)
    
    expected = sum(range(world_size))
    assert data.item() == expected, f"Rank {rank} failed: got {data.item()}, expected {expected}"
    print(f"Rank {rank} data after AllReduce: {data.item()} (Success!)")

if __name__ == "__main__":
    coordinator = DistCoordinator(world_size=2)
    coordinator.run(simple_sum_logic)
