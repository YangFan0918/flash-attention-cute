# flash-attention-cute

基于 CUTLASS/CuTe 模板库从零实现的 FlashAttention-V2 前向 kernel，面向 SM80+（Ampere）架构。

## 特性

- [x] 支持 fp16 / bf16 数据类型
- [x] 支持 head_dim 64 / 128
- [x] 基于 CuTe 的 cp.async 异步拷贝流水线
- [x] Online Softmax 单 pass 计算，softmax scale 融入 exp2f
- [x] Swizzle 消除 shared memory bank conflict
- [x] ldmatrix / MMA 交错调度，隐藏 smem→reg 延迟
- [ ] Grouped Query Attention (GQA)
- [ ] Causal Masking
- [ ] Backward Pass

## 环境要求

- Python >= 3.10
- PyTorch >= 2.0（带 CUDA 支持）
- CUDA Toolkit（支持 SM80+ 架构）
- NVIDIA GPU：Ampere 及以上（如 RTX 3090、A100）
- ninja（加速编译）

## 安装

```bash
# 克隆仓库（含 CUTLASS 子模块）
git clone --recursive https://github.com/YangFan0918/flash-attention-cute.git
cd flash-attention-cute

# 安装
pip install -e .

# 如需运行 benchmark（需要官方 flash-attn 作为对比基准）
pip install -e ".[bench]"
```

> **注意**：默认编译目标为 SM_86（RTX 30 系列）。如需适配其他架构，请修改 `setup.py` 中的 `-gencode` 参数。

## 使用

```python
import torch
from flash_attention import flash_attn_func

batch, seqlen, heads, head_dim = 2, 1024, 8, 64
q = torch.randn(batch, seqlen, heads, head_dim, dtype=torch.float16, device="cuda")
k = torch.randn(batch, seqlen, heads, head_dim, dtype=torch.float16, device="cuda")
v = torch.randn(batch, seqlen, heads, head_dim, dtype=torch.float16, device="cuda")

output = flash_attn_func(q, k, v, causal=False)
```

## 测试

```bash
pytest -m cuda
```

## Benchmark

测试环境：NVIDIA RTX 3090，CUDA 12，PyTorch 2.x，对比基准为 [flash-attn](https://github.com/Dao-AILab/flash-attention) >= 2.5。

```bash
python benchmark/bench.py
```

### 正确性验证

全部 15 组配置通过与官方 flash-attn 的数值对齐验证（max_diff ≤ 2.44e-4）。

### 性能对比

| Config | Ours (ms) | Ours TFLOPS | Official (ms) | Official TFLOPS | Speedup |
|--------|-----------|-------------|---------------|-----------------|---------|
| b=8 s=1024 h=16 d=64 | 0.477 | 72.1 | 0.567 | 60.6 | 1.19x |
| b=8 s=2048 h=16 d=64 | 2.005 | 68.5 | 2.028 | 67.8 | 1.01x |
| b=8 s=4096 h=16 d=64 | 7.939 | 69.3 | 8.050 | 68.3 | 1.01x |
| b=16 s=1024 h=32 d=64 | 2.017 | 68.1 | 2.076 | 66.2 | 1.03x |
| b=16 s=2048 h=32 d=64 | 8.067 | 68.1 | 8.081 | 68.0 | 1.00x |
| b=16 s=4096 h=32 d=64 | 31.819 | 69.1 | 32.414 | 67.8 | 1.02x |
| b=8 s=1024 h=16 d=128 | 1.061 | 64.8 | 1.029 | 66.8 | 0.97x |
| b=8 s=2048 h=16 d=128 | 4.084 | 67.3 | 3.824 | 71.9 | 0.94x |
| b=16 s=1024 h=32 d=128 | 4.197 | 65.5 | 3.902 | 70.5 | 0.93x |
| b=16 s=2048 h=32 d=128 | 16.445 | 66.9 | 15.263 | 72.0 | 0.93x |
| b=32 s=2048 h=32 d=64 | 16.087 | 68.3 | 16.386 | 67.1 | 1.02x |
| b=32 s=4096 h=32 d=64 | 64.168 | 68.5 | 65.393 | 67.3 | 1.02x |
| b=32 s=2048 h=32 d=128 | 33.239 | 66.2 | 30.781 | 71.4 | 0.93x |
| b=32 s=4096 h=32 d=128 | 130.811 | 67.2 | 121.376 | 72.5 | 0.93x |
| b=32 s=4096 h=64 d=128 | 263.609 | 66.7 | 245.119 | 71.8 | 0.93x |

**head_dim=64**：全面持平或略优于官方实现，峰值达 72.1 TFLOPS。

**head_dim=128**：达到官方实现的 93%，约 65-67 TFLOPS。

## 项目结构

```
flash-attention-cute/
├── csrc/
│   ├── flash.h                 # ForwardParams 参数结构体
│   ├── flash_api.cpp           # PyTorch C++ 绑定
│   ├── flash_fwd_traits.h      # Kernel 配置 traits（block size, MMA, layout）
│   ├── flash_fwd_launch.cu     # Kernel 启动与模板分发
│   ├── flash_fwd_kernel.cuh    # 核心 Flash Attention kernel
│   └── static_switch.h         # 编译期分发宏
├── flash_attention/
│   ├── __init__.py
│   └── interface.py            # Python API
├── tests/
│   └── test_flash_attn.py
├── benchmark/
│   └── bench.py
├── 3rd/cutlass/                # CUTLASS 子模块
├── setup.py
└── pyproject.toml
```

## 参考

- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [flash-attention (official)](https://github.com/Dao-AILab/flash-attention)
