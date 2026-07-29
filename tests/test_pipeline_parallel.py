import torch
import torch.nn as nn
from src.distributed.coordinator import DistCoordinator
from src.parallelism.pipeline_parallel import PipelineStage, GPipeSchedule

def pp_validation_logic(rank, world_size):
    torch.manual_seed(42)
    
    batch_size = 8
    hidden_dim = 16
    num_micro_batches = 4
    micro_batch_size = batch_size // num_micro_batches
    
    # 1. Modelo de Referência Sequencial (4 camadas lineares sequenciais)
    # y = L3(L2(L1(L0(x))))
    layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(4)]
    full_model = nn.Sequential(*layers)
    
    input_tensor = torch.randn(batch_size, hidden_dim)
    
    with torch.no_grad():
        expected_output = full_model(input_tensor)

    # 2. Dividir camadas entre Ranks
    # Rank 0 fica com Camadas [0, 1] -> Stage 0
    # Rank 1 fica com Camadas [2, 3] -> Stage 1
    if rank == 0:
        local_layers = nn.Sequential(layers[0], layers[1])
    else:
        local_layers = nn.Sequential(layers[2], layers[3])

    stage = PipelineStage(local_layers, rank, world_size)
    schedule = GPipeSchedule(stage, num_micro_batches=num_micro_batches)

    # 3. Executar Pipeline
    # Formato do micro-batch que o Rank 1 espera receber do Rank 0
    mb_shape = (micro_batch_size, hidden_dim)
    
    with torch.no_grad():
        if rank == 0:
            schedule.step(input_tensor=input_tensor)
            print(f"[Rank 0] Processou e enviou 4 micro-batches.")
        else:
            pipeline_output = schedule.step(micro_batch_shape=mb_shape)
            
            # Validação: Comparar saída final do Rank 1 com o modelo sequencial
            diff = torch.abs(pipeline_output - expected_output).max()
            print(f"[Rank 1] PP Max Diff vs Reference: {diff:.8f}")
            assert diff < 1e-5, f"PP Mismatch! Diff: {diff}"
            
            bubble = schedule.calculate_bubble_ratio(world_size, num_micro_batches)
            print(f"[Rank 1] Pipeline Parallelism Success! (Bubble Ratio: {bubble*100:.1f}%)")

if __name__ == "__main__":
    coordinator = DistCoordinator(world_size=2)
    coordinator.run(pp_validation_logic)
