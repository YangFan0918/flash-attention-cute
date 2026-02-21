import torch
from flash_attention import flash_attn_func

# head_dim=64 和 128 分别测
for head_dim in [64, 128]:
    q = torch.randn(1, 256, 8, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 256, 8, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 256, 8, head_dim, device="cuda", dtype=torch.float16)
    # warmup
    for _ in range(3):
        flash_attn_func(q, k, v)
    torch.cuda.synchronize()
    # profiling target
    out = flash_attn_func(q, k, v)
    torch.cuda.synchronize()
    print(f"head_dim={head_dim} done")
