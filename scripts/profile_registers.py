import torch
from flash_attention import flash_attn_func

for d in [64, 128]:
    q = torch.randn(1, 256, 8, d, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 256, 8, d, dtype=torch.float16, device="cuda")
    v = torch.randn(1, 256, 8, d, dtype=torch.float16, device="cuda")
    # warmup
    for _ in range(3):
        flash_attn_func(q, k, v)
    torch.cuda.synchronize()
    print(f"--- head_dim={d} profiling run ---")
    flash_attn_func(q, k, v)
    torch.cuda.synchronize()
