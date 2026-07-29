import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List
from src.parallelism.micro_batch import split_batch, merge_batches

class PipelineStage(nn.Module):
    """Representa um estágio do pipeline (um subconjunto de camadas do modelo)."""
    def __init__(self, module: nn.Module, rank: int, world_size: int):
        super().__init__()
        self.module = module
        self.rank = rank
        self.world_size = world_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)

class GPipeSchedule:
    """
    Agendador GPipe (Fill-Drain) para inferência.
    Rank 0 processa e envia para o Rank 1, que recebe, processa e finaliza.
    """
    def __init__(self, stage: PipelineStage, num_micro_batches: int):
        self.stage = stage
        self.num_micro_batches = num_micro_batches
        self.rank = stage.rank
        self.world_size = stage.world_size

    def step(self, input_tensor: torch.Tensor = None, micro_batch_shape: tuple = None) -> torch.Tensor:
        micro_inputs = []
        
        # 1. Se for o primeiro estágio (Rank 0), divide a entrada principal
        if self.rank == 0:
            assert input_tensor is not None, "Rank 0 precisa de um input_tensor!"
            micro_inputs = split_batch(input_tensor, self.num_micro_batches)
        
        micro_outputs = []

        for i in range(self.num_micro_batches):
            # 2. Receber do rank anterior (se não for Rank 0)
            if self.rank > 0:
                # Aloca buffer vazio para receber o micro-batch
                recv_buffer = torch.empty(micro_batch_shape, dtype=torch.float32)
                dist.recv(recv_buffer, src=self.rank - 1)
                mb_input = recv_buffer
            else:
                mb_input = micro_inputs[i]

            # 3. Computação do Estágio Local
            mb_output = self.stage(mb_input)

            # 4. Enviar para o próximo rank (se não for o último estágio)
            if self.rank < self.world_size - 1:
                dist.send(mb_output, dst=self.rank + 1)
            else:
                micro_outputs.append(mb_output)

        # 5. Se for o último estágio, junta todos os micro-outputs e retorna
        if self.rank == self.world_size - 1:
            return merge_batches(micro_outputs)
        return None

    @staticmethod
    def calculate_bubble_ratio(num_stages: int, num_micro_batches: int) -> float:
        """Calcula a proporção teórica de tempo ocioso (bubble ratio) do GPipe."""
        return (num_stages - 1) / (num_micro_batches + num_stages - 1)
