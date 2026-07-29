from dataclasses import dataclass
import os

@dataclass
class DistConfig:
    world_size: int = 2
    rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: str = "29500"
    backend: str = "gloo"  # Gloo é perfeito para simular multi-gpu em 1 GPU ou CPU

    @classmethod
    def from_env(cls):
        return cls(
            world_size=int(os.environ.get("WORLD_SIZE", 2)),
            rank=int(os.environ.get("RANK", 0)),
        )
