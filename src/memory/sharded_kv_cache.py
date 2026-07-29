import torch
from typing import Tuple

class ShardedKVCache:
    """
    Armazena apenas um subconjunto das heads de atenção (shards de K e V).
    Simula a economia de memória em sistemas distribuídos.
    """
    def __init__(self, num_heads: int, head_dim: int, world_size: int, rank: int):
        self.rank = rank
        self.world_size = world_size
        self.head_dim = head_dim
        
        # Cada rank cuida apenas de uma fração das heads
        assert num_heads % world_size == 0, "Num heads deve ser divisível pelo world_size"
        self.num_local_heads = num_heads // world_size
        
        # Inicializa caches vazios: [batch, heads, seq_len, dim]
        self.k_cache = torch.empty(0)
        self.v_cache = torch.empty(0)

    def update(self, k_shard: torch.Tensor, v_shard: torch.Tensor):
        """
        Adiciona novos tokens ao cache local.
        k_shard/v_shard: [batch, local_heads, new_tokens, head_dim]
        """
        if self.k_cache.numel() == 0:
            self.k_cache = k_shard
            self.v_cache = v_shard
        else:
            # Concatena na dimensão da sequência (dim 2)
            self.k_cache = torch.cat([self.k_cache, k_shard], dim=2)
            self.v_cache = torch.cat([self.v_cache, v_shard], dim=2)

    def get_shards(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k_cache, self.v_cache

    @property
    def memory_usage_bytes(self) -> int:
        """Retorna o tamanho do cache local em bytes."""
        return self.k_cache.element_size() * self.k_cache.nelement() + \
               self.v_cache.element_size() * self.v_cache.nelement()
