import torch
from src.distributed.coordinator import DistCoordinator
from src.profiling.comm_profiler import CommProfiler
from src.profiling.cost_model import AnalyticalCostModel

def profiling_logic(rank, world_size):
    profiler = CommProfiler()
    # Parâmetros estimados para sua máquina (podem variar)
    # Alpha: 0.1ms latência | Beta: 5 GB/s bandwidth local
    model = AnalyticalCostModel(alpha_ms=0.1, beta_gb_s=5.0)

    # Testar diferentes tamanhos de tensor
    # 1M, 10M, 50M de elementos FP32 (4 bytes cada)
    sizes = [10**6, 10**7, 25 * 10**6] 
    
    if rank == 0:
        print(f"{'Size (MB)':<10} | {'Measured (ms)':<12} | {'Predicted (ms)':<12} | {'Error %':<10}")
        print("-" * 55)

    for s in sizes:
        tensor = torch.randn(s)
        num_bytes = tensor.nelement() * tensor.element_size()
        
        # Medição Real
        profiler.profile_all_reduce(tensor)
        measured_ms = profiler.events[-1].duration_ms
        
        # Predição do Modelo
        predicted_ms = model.predict_ms(num_bytes)
        
        error = abs(measured_ms - predicted_ms) / measured_ms * 100
        
        if rank == 0:
            size_mb = num_bytes / 1024**2
            print(f"{size_mb:>10.2f} | {measured_ms:>12.2f} | {predicted_ms:>12.2f} | {error:>10.2f}%")

if __name__ == "__main__":
    coordinator = DistCoordinator(world_size=2)
    coordinator.run(profiling_logic)
