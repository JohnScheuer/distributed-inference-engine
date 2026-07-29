# DESIGN.md — distributed-inference-engine

## Architecture Overview
This engine simulates a multi-GPU environment using a Multi-Process architecture synchronized via `torch.distributed`. It implements the core primitives of Large Language Model (LLM) scaling.

### System Components
1. **Coordinator:** Orchestrates the lifecycle of distributed workers using `mp.spawn`.
2. **Workers:** Independent Python processes representing virtual GPUs.
3. **Communication Layer:** Uses the **Gloo** backend for cross-process synchronization (AllReduce, Send/Recv).

---

## Parallelism Strategies

### 1. Tensor Parallelism (TP)
- **ColumnParallelLinear:** Shards the weight matrix across output features. Each worker computes a partial output vector.
- **RowParallelLinear:** Shards the weight matrix across input features. Workers compute partial sums and synchronize via **AllReduce** to produce the final result.
- **Application:** Used to distribute large MLP and Attention blocks.

### 2. Pipeline Parallelism (PP)
- **Layer Sharding:** Model layers are divided into stages (e.g., Rank 0 handles layers 0-15).
- **Micro-batching:** To minimize the "Pipeline Bubble," batches are split into smaller units. We implemented a **GPipe-style filling scheduler**.

### 3. KV Cache Sharding
- **Memory Optimization:** Instead of replicating the KV Cache, each rank only stores the Heads (Attention) it is responsible for.
- **Memory Scaling:** VRAM usage per rank scales as `Total_KV / World_Size`.

---

## Technical Decisions
- **Backend:** `gloo` was chosen over NCCL to allow for reliable multi-process simulation on a single GPU or CPU environment.
- **Numerical Fidelity:** All parallel layers were validated against single-process baselines to ensure a 0.00 max absolute error.
- **Cost Modeling:** We implemented an analytical model (`T = alpha + B/beta`) to predict communication overhead, revealing the bandwidth limitations of process-level simulation compared to NVLink.

---

## Limitations
- **Simulation Overhead:** Inter-process communication on a single machine is slower than physical multi-GPU interconnects.
- **Static Partitioning:** Layers are divided evenly; dynamic load balancing is not yet implemented.
