import torch
import torch.nn.functional as F
import time


def benchmark_fn(fn, *args, warmup=10, repeat=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeat * 1000  # ms


def main():
    batch, seqlen, heads, head_dim = 8, 2048, 16, 64
    dtype = torch.float16
    device = "cuda"

    q = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype)

    # PyTorch reference
    qt, kt, vt = [x.transpose(1, 2) for x in (q, k, v)]
    pytorch_ms = benchmark_fn(
        lambda *a: F.scaled_dot_product_attention(*a, is_causal=False), qt, kt, vt
    )
    print(f"PyTorch SDPA:         {pytorch_ms:.3f} ms")

    # Our implementation
    from flash_attention import flash_attn_func
    ours_ms = benchmark_fn(flash_attn_func, q, k, v, False)
    print(f"Flash Attention CuTe: {ours_ms:.3f} ms")


if __name__ == "__main__":
    main()
