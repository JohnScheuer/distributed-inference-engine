import torch
import torch.nn as nn
from src.distributed.coordinator import DistCoordinator
from src.parallelism.tensor_parallel import ColumnParallelLinear, RowParallelLinear

def tp_validation_logic(rank, world_size):
    torch.manual_seed(42)
    
    batch_size = 2
    in_dim = 8
    out_dim = 4
    
    # 1. Referência
    ref_linear = nn.Linear(in_dim, out_dim)
    input_tensor = torch.randn(batch_size, in_dim)
    expected_output = ref_linear(input_tensor)

    # 2. ColumnParallel
    tp_col = ColumnParallelLinear(in_dim, out_dim, world_size, rank)
    start = rank * (out_dim // world_size)
    end = (rank + 1) * (out_dim // world_size)
    
    with torch.no_grad():
        tp_col.weight.copy_(ref_linear.weight[start:end, :])
        tp_col.bias.copy_(ref_linear.bias[start:end])
    
    # 3. RowParallel
    tp_row = RowParallelLinear(in_dim, out_dim, world_size, rank)
    start_in = rank * (in_dim // world_size)
    end_in = (rank + 1) * (in_dim // world_size)
    
    with torch.no_grad():
        tp_row.weight.copy_(ref_linear.weight[:, start_in:end_in])
        # CÓPIA CORRETA: Bias completo em todos os ranks
        tp_row.bias.copy_(ref_linear.bias) 
    
    # Execução
    input_shard = input_tensor[:, start_in:end_in]
    row_out = tp_row(input_shard)

    # Validação
    diff = torch.abs(row_out - expected_output).max()
    print(f"[Rank {rank}] RowParallel Max Diff: {diff:.8f}")
    assert diff < 1e-5, f"Rank {rank} TP mismatch! Diff: {diff}"
    print(f"[Rank {rank}] TP Validation Success!")

if __name__ == "__main__":
    coordinator = DistCoordinator(world_size=2)
    coordinator.run(tp_validation_logic)
