import torch
import torch.distributed as dist
import time
from dataclasses import dataclass
from typing import List

@dataclass
class CommEvent:
    op_name: str
    num_bytes: int
    duration_ms: float

class CommProfiler:
    """Registra eventos de comunicação para análise de custo."""
    def __init__(self):
        self.events: List[CommEvent] = []

    def profile_all_reduce(self, tensor: torch.Tensor):
        num_bytes = tensor.nelement() * tensor.element_size()
        
        # Sincroniza a GPU/CPU antes de começar
        if tensor.is_cuda: torch.cuda.synchronize()
        
        start = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        if tensor.is_cuda: torch.cuda.synchronize()
        end = time.perf_counter()
        
        duration_ms = (end - start) * 1000
        self.events.append(CommEvent("all_reduce", num_bytes, duration_ms))
        return tensor

    def report(self):
        print(f"\n{'Op':<12} | {'Size (MB)':<10} | {'Time (ms)':<10} | {'Bandwidth (GB/s)':<10}")
        print("-" * 55)
        for e in self.events:
            size_mb = e.num_bytes / (1024**2)
            # Bandwidth: (Bytes / 10^9) / (Seconds)
            bw = (e.num_bytes / 1e9) / (e.duration_ms / 1000)
            print(f"{e.op_name:<12} | {size_mb:>10.2f} | {e.duration_ms:>10.2f} | {bw:>10.2f}")
