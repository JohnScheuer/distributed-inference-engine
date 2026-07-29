import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.distributed.coordinator import DistCoordinator
from src.parallelism.tensor_parallel import ColumnParallelLinear, RowParallelLinear
import time

def e2e_inference_logic(rank, world_size):
    model_id = "Qwen/Qwen2-0.5B-Instruct"
    
    print(f"[Rank {rank}] Loading weights...")
    full_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Pegamos o bloco MLP da Camada 0
    ref_mlp = full_model.model.layers[0].mlp
    
    in_dim = ref_mlp.gate_proj.in_features
    inter_dim = ref_mlp.gate_proj.out_features
    
    # Inicializamos nossos shards
    gate_shard = ColumnParallelLinear(in_dim, inter_dim, world_size, rank)
    up_shard = ColumnParallelLinear(in_dim, inter_dim, world_size, rank)
    down_shard = RowParallelLinear(inter_dim, in_dim, world_size, rank)

    # Fatiamento dos pesos
    shard_slice = slice(rank * (inter_dim // world_size), (rank + 1) * (inter_dim // world_size))
    
    with torch.no_grad():
        # Cópia do Gate Proj (Sem bias no Qwen2)
        gate_shard.weight.copy_(ref_mlp.gate_proj.weight[shard_slice, :])
        if ref_mlp.gate_proj.bias is not None:
            gate_shard.bias.copy_(ref_mlp.gate_proj.bias[shard_slice])
        else:
            gate_shard.bias.fill_(0) # Zera se não existir

        # Cópia do Up Proj
        up_shard.weight.copy_(ref_mlp.up_proj.weight[shard_slice, :])
        if ref_mlp.up_proj.bias is not None:
            up_shard.bias.copy_(ref_mlp.up_proj.bias[shard_slice])
        else:
            up_shard.bias.fill_(0)

        # Cópia do Down Proj (Row Parallel)
        down_shard.weight.copy_(ref_mlp.down_proj.weight[:, shard_slice])
        if ref_mlp.down_proj.bias is not None:
            down_shard.bias.copy_(ref_mlp.down_proj.bias)
        else:
            down_shard.bias.fill_(0)

    # 3. Executar Inferência
    prompt = "Distributed systems are"
    inputs = tokenizer(prompt, return_tensors="pt")
    x = full_model.model.embed_tokens(inputs.input_ids)

    # Warmup
    for _ in range(3):
        _ = down_shard(torch.nn.functional.silu(gate_shard(x)) * up_shard(x))

    # Benchmark
    start_time = time.perf_counter()
    with torch.no_grad():
        out_parallel = down_shard(torch.nn.functional.silu(gate_shard(x)) * up_shard(x))
    end_time = time.perf_counter()
    
    # 4. Comparar com Referência
    with torch.no_grad():
        out_ref = ref_mlp(x)
    
    diff = torch.abs(out_parallel - out_ref).max()
    
    if rank == 0:
        print("\n" + "="*50)
        print(f"E2E DISTRIBUTED MLP SUCCESS (Qwen2-0.5B)")
        print("="*50)
        print(f"Max Difference: {diff:.8f}")
        print(f"Inference Time: {(end_time - start_time)*1000:.2f} ms")
        print(f"Numerical Fidelity: {'PASS' if diff < 1e-5 else 'FAIL'}")

if __name__ == "__main__":
    coordinator = DistCoordinator(world_size=2)
    coordinator.run(e2e_inference_logic)
