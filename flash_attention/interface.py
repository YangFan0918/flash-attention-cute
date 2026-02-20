import torch
from flash_attention._C import forward


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """
    Flash Attention v2 forward pass.

    Args:
        q: (batch, seqlen_q, heads, head_dim), fp16/bf16
        k: (batch, seqlen_k, heads_k, head_dim), fp16/bf16
        v: (batch, seqlen_k, heads_k, head_dim), fp16/bf16
        causal: whether to apply causal mask

    Returns:
        o: (batch, seqlen_q, heads, head_dim)
    """
    return forward(q, k, v, causal)
