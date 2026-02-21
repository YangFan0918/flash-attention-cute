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
        (16, 1024, 32, 64),
        (16, 2048, 32, 64),
        (16, 4096, 32, 64),
        (8, 1024, 16, 128),
        (8, 2048, 16, 128),
        (16, 1024, 32, 128),
        (16, 2048, 32, 128),
        (32, 2048, 32, 64),
        (32, 4096, 32, 64),
        (32, 2048, 32, 128),
        (32, 4096, 32, 128),
        (32, 4096, 64, 128),
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

    # ==================== Correctness check ====================
    if has_official:
        print("Correctness check (vs official flash-attn):")
        all_pass = True
        for batch, seqlen, heads, head_dim in configs:
            q = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype)
            k = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype)
            v = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype)
            out = flash_attn_func(q, k, v, False)
            ref = flash_attn_official(q, k, v, causal=False)
            max_diff = (out - ref).abs().max().item()
            mean_diff = (out - ref).abs().mean().item()
            passed = max_diff < 1e-2
            all_pass = all_pass and passed
            label = f"b={batch} s={seqlen} h={heads} d={head_dim}"
            status = "PASS" if passed else "FAIL"
            print(f"  {label:<30} {status}  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")
        print("All passed.\n" if all_pass else "\nSome tests FAILED!\n")

    # ==================== Benchmark ====================
    header = f"{'Config':<30} {'Ours (ms)':>10} {'Ours TFLOPS':>12} {'Official (ms)':>14} {'Off TFLOPS':>11} {'Speedup':>10}"
    print(header)
    print("-" * len(header))

    for batch, seqlen, heads, head_dim in configs:
        q = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype)

        # 4 * N * d per query-key-value: 2 for Q@K^T + 2 for scores@V
        flops = 4 * batch * heads * seqlen * seqlen * head_dim

        ours_ms = benchmark_fn(flash_attn_func, q, k, v, False)
        ours_tflops = flops / (ours_ms * 1e-3) / 1e12

        label = f"b={batch} s={seqlen} h={heads} d={head_dim}"
        if has_official:
            official_ms = benchmark_fn(lambda q, k, v, c: flash_attn_official(q, k, v, causal=c), q, k, v, False)
            off_tflops = flops / (official_ms * 1e-3) / 1e12
            speedup = official_ms / ours_ms
            print(f"{label:<30} {ours_ms:>10.3f} {ours_tflops:>11.1f} {official_ms:>14.3f} {off_tflops:>11.1f} {speedup:>9.2f}x")
        else:
            print(f"{label:<30} {ours_ms:>10.3f} {ours_tflops:>11.1f} {'N/A':>14} {'N/A':>11} {'N/A':>10}")


if __name__ == "__main__":
    main()
