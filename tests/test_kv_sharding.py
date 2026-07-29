import torch
import torch.nn.functional as F
from src.distributed.coordinator import DistCoordinator
from src.distributed.comm import all_reduce_sum
from src.memory.sharded_kv_cache import ShardedKVCache

def kv_sharding_logic(rank, world_size):
    torch.manual_seed(42)
    
    batch = 1
    num_heads = 4
    seq_len = 8
    head_dim = 16
    
    # 1. Gerar K, V globais para referência
    k_global = torch.randn(batch, num_heads, seq_len, head_dim)
    v_global = torch.randn(batch, num_heads, seq_len, head_dim)
    q_global = torch.randn(batch, num_heads, 1, head_dim) # Query de 1 token (decode)

    # 2. Referência: Atenção Standard (Single-GPU)
    # Scaled Dot-Product Attention
    attn_weights = torch.matmul(q_global, k_global.transpose(-1, -2)) / (head_dim ** 0.5)
    attn_probs = F.softmax(attn_weights, dim=-1)
    ref_output = torch.matmul(attn_probs, v_global) # [batch, heads, 1, dim]

    # 3. Simular KV Sharding (Multi-GPU)
    local_kv = ShardedKVCache(num_heads, head_dim, world_size, rank)
    
    # Cada rank pega seu pedaço das heads
    heads_per_rank = num_heads // world_size
    start_head = rank * heads_per_rank
    end_head = (rank + 1) * heads_per_rank
    
    k_shard = k_global[:, start_head:end_head, :, :]
    v_shard = v_global[:, start_head:end_head, :, :]
    q_shard = q_global[:, start_head:end_head, :, :]
    
    local_kv.update(k_shard, v_shard)
    
    # Atenção Local (Apenas as heads deste rank)
    local_k, local_v = local_kv.get_shards()
    local_attn_weights = torch.matmul(q_shard, local_k.transpose(-1, -2)) / (head_dim ** 0.5)
    local_attn_probs = F.softmax(local_attn_weights, dim=-1)
    local_output_shard = torch.matmul(local_attn_probs, local_v)

    # 4. Reconstruir o output Global
    # Em Tensor Parallelism real, as saídas das heads são concatenadas.
    # Mas como o RowParallelLinear viria depois, simularemos a reconstrução.
    # Vamos usar um buffer global e colocar nossos shards lá
    reconstructed_output = torch.zeros_like(ref_output)
    reconstructed_output[:, start_head:end_head, :, :] = local_output_shard
    
    # Sincroniza todos os pedaços entre os workers
    all_reduce_sum(reconstructed_output)

    # Validação
    diff = torch.abs(reconstructed_output - ref_output).max()
    mem_mb = local_kv.memory_usage_bytes / 1024**2
    
    print(f"[Rank {rank}] Memory usage: {mem_mb:.4f} MB")
    print(f"[Rank {rank}] Attention Max Diff: {diff:.8f}")
    
    assert diff < 1e-5, f"KV Sharding mismatch on Rank {rank}!"
    if rank == 0:
        print(f"KV Cache Sharding validated! Total VRAM saved: {mem_mb * world_size:.2f}MB total vs {mem_mb:.2f}MB per worker.")

if __name__ == "__main__":
    coordinator = DistCoordinator(world_size=2)
    coordinator.run(kv_sharding_logic)
