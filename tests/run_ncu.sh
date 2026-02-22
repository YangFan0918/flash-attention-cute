#!/bin/bash
# Usage: bash tests/run_ncu.sh [64|128]
# Output: ncu_report_d${D}.ncu-rep (open in Nsight Compute GUI)

D=${1:-128}

PYTHONPATH=$(pwd) ncu \
  --set full \
  --kernel-name "flash_fwd_kernel" \
  --launch-skip 3 \
  --launch-count 1 \
  -o ncu_report_d${D} \
  python -c "
import torch
from flash_attention import flash_attn_func
d = ${D}
q = torch.randn(2, 2048, 16, d, dtype=torch.float16, device='cuda')
k = torch.randn(2, 2048, 16, d, dtype=torch.float16, device='cuda')
v = torch.randn(2, 2048, 16, d, dtype=torch.float16, device='cuda')
for _ in range(3):
    flash_attn_func(q, k, v)
torch.cuda.synchronize()
flash_attn_func(q, k, v)
torch.cuda.synchronize()
"
