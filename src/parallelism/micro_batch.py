import torch
from typing import List

def split_batch(tensor: torch.Tensor, num_micro_batches: int) -> List[torch.Tensor]:
    """Divide um lote (batch) em múltiplos micro-lotes ao longo da dimensão 0."""
    return list(torch.chunk(tensor, num_micro_batches, dim=0))

def merge_batches(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Concatena os micro-lotes de volta em um único tensor."""
    return torch.cat(tensors, dim=0)
