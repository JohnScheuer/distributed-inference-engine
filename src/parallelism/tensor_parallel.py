import torch
import torch.nn as nn
import torch.distributed as dist
from src.distributed.comm import all_reduce_sum

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.shard_size = out_features // world_size
        
        self.weight = nn.Parameter(torch.randn(self.shard_size, in_features))
        self.bias = nn.Parameter(torch.randn(self.shard_size))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)

class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.shard_size = in_features // world_size
        
        self.weight = nn.Parameter(torch.randn(out_features, self.shard_size))
        # No RowParallel, o bias completo fica em todos os ranks
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x_shard):
        # 1. Multiplicação local (cada rank computa uma parte da soma)
        # x_shard: [batch, in_shard], weight: [out, in_shard]
        partial_output = torch.nn.functional.linear(x_shard, self.weight)
        
        # 2. AllReduce: soma as contribuições de todos os ranks
        all_reduce_sum(partial_output)
        
        # 3. Adiciona o bias completo (já que partial_output agora é a soma total)
        return partial_output + self.bias
