<div align="center">

# distributed-inference-engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Distributed](https://img.shields.io/badge/PyTorch-Distributed-orange.svg)](https://pytorch.org/docs/stable/distributed.html)

**Simulated distributed inference runtime for algorithmic validation of Tensor Parallelism and Pipeline Parallelism.**

Validated on Qwen2-0.5B | Megatron-style TP | GPipe PP | KV Cache Sharding

</div>

---

## What This Does

This project implements the core infrastructure required to scale Large Language Models across multiple compute nodes. It uses `torch.distributed` with the **Gloo backend** to simulate a multi-GPU environment via independent Python processes.

**Objective:** Validate the mathematical correctness of parallelism algorithms (sharding, synchronization, scheduling) on a single machine before multi-GPU deployment.

> **Note:** This validates parallelism **logic**, not hardware performance. Production multi-GPU scaling requires the NCCL backend and physical interconnects (NVLink/PCIe).

---

## Performance & Fidelity (Qwen2-0.5B)

**Configuration:** 2-Rank Parallelism, Gloo (CPU Sim), FP32 Precision.  
**Hardware:** NVIDIA RTX 2070 8GB | Intel i7-10700K.

### 1. MLP Block Tensor Parallelism (TP)
Validating `ColumnParallelLinear` + `RowParallelLinear` on a real Qwen2 MLP block (`hidden_size=896`).

| Config | Latency | Max Error vs Baseline |
| :--- | :--- | :--- |
| Single-process Baseline | ~2.10 ms | Reference |
| **2-Rank TP (Simulated)** | **3.74 ms** | **0.00000000** |
| Communication Overhead | ~78% | - |

*Note: High overhead is expected in simulation as inter-process communication via RAM (Gloo) is significantly slower than GPU compute.*

### 2. Pipeline Parallelism (PP)
| Metric | Result |
| :--- | :--- |
| **Bubble Ratio (Analytic)** | **20%** (2 stages, 4 micro-batches) |
| **Micro-batch Sync** | Bit-identical fidelity vs Sequential |

### 3. KV Cache Sharding
Memory footprint per node scales as `1/world_size`:
- **2 Ranks:** 50% memory saving per node.
- **4 Ranks:** 75% memory saving per node.
- **8 Ranks:** 87.5% memory saving per node.

---

## Features

- **Megatron-style TP:** Manual implementation of Row and Column parallel layers with AllReduce synchronization.
- **GPipe Schedule:** Fill-drain micro-batch scheduling to optimize pipeline stage utilization.
- **KV Cache Head Sharding:** Distributed attention heads to enable scaling beyond single-GPU VRAM limits.
- **Bit-identical Fidelity:** Validated on Qwen2-0.5B real weights with zero numerical divergence in FP32.

---

## Limitations

- **CPU Simulation:** Uses Gloo backend; not intended for NCCL/multi-GPU performance testing.
- **Single-Machine:** Multi-process only, not optimized for cross-node networking.
- **FP32 Only:** FP16 is not used to avoid hardware-specific rounding differences during validation.
- **Scheduling:** Implements GPipe; advanced schedules like 1F1B are not covered.

---

## What This Does NOT Do

- **Physical Multi-GPU Execution:** Does not require multiple GPUs to run.
- **Production Latency Optimization:** Overhead reflects OS process sync, not NVLink speed.
- **Data Parallelism (DP):** Focused exclusively on Model Parallelism (TP/PP).

---

## LLM Systems Portfolio

This project is part of a comprehensive LLM inference portfolio covering kernels, compilers, serving, deployment, and distributed systems.

**Latest additions to the series:**
- [distributed-inference-engine](https://github.com/JohnScheuer/distributed-inference-engine): Parallelism & Scaling.
- [quantization-runtime](https://github.com/JohnScheuer/quantization-runtime): 4-bit AWQ/GPTQ Compression.
- [rag-inference-stack](https://github.com/JohnScheuer/rag-inference-stack): Knowledge retrieval & API.
- [lora-inference-runtime](https://github.com/JohnScheuer/lora-inference-runtime): Multi-tenant serving.
- [speculative-decoding-runtime](https://github.com/JohnScheuer/speculative-decoding-runtime): Latency acceleration.

Full portfolio: [github.com/JohnScheuer](https://github.com/JohnScheuer)

---

## License

[MIT](LICENSE) - Joao Felipe De Souza, 2026
