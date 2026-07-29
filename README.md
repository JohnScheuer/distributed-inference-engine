<div align="center">

# distributed-inference-engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Distributed](https://img.shields.io/badge/PyTorch-Distributed-orange.svg)](https://pytorch.org/docs/stable/distributed.html)
[![Status](https://img.shields.io/badge/Status-Numerical%20Fidelity%20Passed-brightgreen.svg)](#)

**Simulated Multi-GPU inference runtime implementing Megatron-style Tensor Parallelism and GPipe Pipeline Parallelism from scratch.**

TP & PP Implementation | AllReduce Sync | KV Cache Sharding | Bit-identical Fidelity

</div>

---

## What It Does

This project implements the core infrastructure required to scale Large Language Models across multiple compute nodes. It breaks down monolithic layers into sharded components, managing cross-process communication and micro-batch scheduling manually via `torch.distributed`.

- **Tensor Parallelism (TP):** Megatron-style weight sharding across rows and columns.
- **Pipeline Parallelism (PP):** Inter-layer distribution with a GPipe-style scheduler.
- **KV Cache Sharding:** Attention head distribution to optimize memory footprint.

---

## Performance & Fidelity (Qwen2-0.5B)

**Configuration:** 2-Rank Parallelism, Gloo Backend (CPU Sim), FP32 Precision.

| Component | Metric | Result |
| :--- | :--- | :--- |
| **MLP Block (TP)** | Max Error vs Baseline | **0.00000000** |
| **MLP Block (TP)** | Distributed Latency | **3.74 ms** |
| **Pipeline (PP)** | Bubble Ratio (2 stages/4 MB) | 20% |
| **KV Cache** | Memory Saving per Node | **~50%** |
| **E2E (Qwen2)** | 2-Rank TP Correctness | **Bit-identical** |

> **Numerical Note:** Zero absolute error is expected in FP32 with the Gloo backend due to deterministic CPU operations. Production NCCL on physical Multi-GPU environments typically introduces FP16 rounding differences (< 1e-3).

---

## Features

- **Megatron-style TP:** Manual implementation of `ColumnParallelLinear` and `RowParallelLinear` with AllReduce synchronization (no NCCL dependency required for validation).
- **GPipe Pipeline Schedule:** Fill-drain micro-batch scheduling logic with configurable stage counts and theoretical bubble ratio modeling.
- **KV Cache Head Sharding:** Attention heads distributed across ranks, reducing per-node VRAM consumption by 1/world_size.
- **Real-weight Validation:** TP-sharded MLP forward pass on Qwen2-0.5B weights (`hidden_size=896`, `intermediate_size=4864`) produces bit-identical output to single-process baseline.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Tensor Parallelism Validation
export PYTHONPATH=.
python tests/test_tensor_parallel.py

# 3. Run E2E Distributed MLP (Qwen2-0.5B)
python benchmarks/e2e_distributed.py
```

---

## LLM Systems Portfolio

This is the final piece of a 5-project series on LLM inference systems:
- [distributed-inference-engine](https://github.com/JohnScheuer/distributed-inference-engine): Parallelism & Scaling.
- [quantization-runtime](https://github.com/JohnScheuer/quantization-runtime): 4-bit AWQ/GPTQ Compression.
- [rag-inference-stack](https://github.com/JohnScheuer/rag-inference-stack): Knowledge retrieval & API.
- [lora-inference-runtime](https://github.com/JohnScheuer/lora-inference-runtime): Multi-tenant serving.
- [speculative-decoding-runtime](https://github.com/JohnScheuer/speculative-decoding-runtime): Latency acceleration.

---

## License

[MIT](LICENSE) - Joao Felipe De Souza, 2026
