<div align="center">

# distributed-inference-engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Distributed](https://img.shields.io/badge/PyTorch-Distributed-orange.svg)](https://pytorch.org/docs/stable/distributed.html)
[![Status](https://img.shields.io/badge/Status-Numerical%20Fidelity%20Passed-brightgreen.svg)](#)

**Simulated Multi-GPU inference runtime implementing Tensor and Pipeline Parallelism from scratch.**

TP & PP Implementation | AllReduce Sync | KV Cache Sharding | 0.00 Numerical Error

</div>

---

## What It Does

This project implements the core infrastructure required to serve Large Language Models across multiple GPUs. It breaks down monolithic models into shards, managing communication and synchronization manually via `torch.distributed`.

- **Tensor Parallelism:** Splitting matrix multiplications (Megatron-LM style).
- **Pipeline Parallelism:** Distributing layers across stages with micro-batching.
- **KV Cache Sharding:** Reducing memory footprint by sharding attention heads.

---

## Performance & Fidelity (Qwen2-0.5B)

**Configuration:** 2-Rank Parallelism, Gloo Backend, FP32 Precision.

| Component | Metric | Result |
| :--- | :--- | :--- |
| **MLP Block** | Max Difference vs Baseline | **0.00000000** |
| **MLP Block** | Distributed Latency | **3.74 ms** |
| **Pipeline** | Bubble Ratio (2 stages/4 MB) | 20% |
| **KV Cache** | Memory Saving per Node | **~50%** |

---

## Features

- **Standard Collective Ops:** Manual implementation of Row and Column parallel layers.
- **Micro-batch Scheduling:** GPipe-style fill-drain schedule to optimize pipeline throughput.
- **Communication Profiling:** Real vs. Predicted bandwidth analysis.
- **Architectural Validation:** Successfully tested with Qwen2-0.5B real weights.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Tensor Parallelism Validation
export PYTHONPATH=.
python tests/test_tensor_parallel.py

# 3. Run E2E Distributed MLP (Qwen2)
python benchmarks/e2e_distributed.py
```

---

## Design Decisions

- **Numerical Stability:** Chose the Gloo backend for local process simulation, ensuring identical results to single-GPU execution.
- **Bias Handling:** Implemented proper post-AllReduce bias addition to maintain mathematical consistency in parallel layers.
- **Isolated Groups:** Designed for scalability beyond 2 ranks.

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
