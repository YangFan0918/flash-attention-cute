import pytest
import torch
import torch.nn.functional as F


def reference_attn(q, k, v, causal=False):
    """PyTorch reference: scaled dot-product attention."""
    scale = q.shape[-1] ** -0.5
    # (batch, heads, seqlen, head_dim)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
    return out.transpose(1, 2)


@pytest.mark.cuda
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("seqlen", [128, 256, 512])
@pytest.mark.parametrize("heads", [8])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_flash_attn(batch, seqlen, heads, head_dim, causal, dtype):
    from flash_attention import flash_attn_func

    torch.manual_seed(42)
    q = torch.randn(batch, seqlen, heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen, heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen, heads, head_dim, device="cuda", dtype=dtype)

    out = flash_attn_func(q, k, v, causal=causal)
    ref = reference_attn(q, k, v, causal=causal)

    assert torch.allclose(out, ref, atol=1e-2, rtol=1e-3), \
        f"max diff: {(out - ref).abs().max().item()}"
