import torch
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
    configs = [
        (8, 1024, 16, 64),
        (8, 2048, 16, 64),
        (8, 4096, 16, 64),
        (8, 1024, 16, 128),
        (8, 2048, 16, 128),
    ]
    dtype = torch.float16
    device = "cuda"

    from flash_attention import flash_attn_func
    try:
        from flash_attn import flash_attn_func as flash_attn_official
        has_official = True
    except ImportError:
        has_official = False
        print("flash-attn not installed, skipping official comparison")
        print("Install with: pip install flash-attn --no-build-isolation\n")

    print(f"{'Config':<30} {'Ours (ms)':>10} {'Official (ms)':>14} {'Speedup':>10}")
    print("-" * 68)

    for batch, seqlen, heads, head_dim in configs:
        q = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype)

        ours_ms = benchmark_fn(flash_attn_func, q, k, v, False)

        label = f"b={batch} s={seqlen} h={heads} d={head_dim}"
        if has_official:
            official_ms = benchmark_fn(flash_attn_official, q, k, v, causal=False)
            speedup = official_ms / ours_ms
            print(f"{label:<30} {ours_ms:>10.3f} {official_ms:>14.3f} {speedup:>9.2f}x")
        else:
            print(f"{label:<30} {ours_ms:>10.3f} {'N/A':>14} {'N/A':>10}")


if __name__ == "__main__":
    main()
