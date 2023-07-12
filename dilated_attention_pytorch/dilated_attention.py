from typing import Optional, Sequence, Tuple, Union

import torch
from einops import rearrange
from torch import Tensor, nn

import xformers.ops as xops


class DilatedAttention(nn.Module):
    """Implement dilated, scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(
        self,
        segment_lengths: Sequence[int],
        dilation_rates: Sequence[int],
        softmax_scale: Optional[float] = None,
        attention_dropout: float = 0.0,
        op: Optional[xops.AttentionOp] = None,
    ):
        super().__init__()
        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.op = op

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, causal: bool = False
    ) -> Tensor:
        # Notation:
        #   b - batch size
        #   n - sequence length
        #   h - number of heads
        #   d - embedding dimension
        #   s - segment length
        #   r - dilation rate
        #
        # Input shape of query, key, value: (b, n, h, d)
        b, _, h, _ = query.shape
        out = torch.zeros_like(query)

        # TODO: Move the assertions to higher-level class, at initialization time.
        assert len(self.segment_lengths) == len(self.dilation_rates)
        num_groups = len(self.dilation_rates)
        assert h % num_groups == 0
        group_size = h // num_groups

        for i, (r, s) in enumerate(zip(self.dilation_rates, self.segment_lengths)):
            # Split the input sequences into segments of length 'self.segment_length'
            q = rearrange(query, "b (n s) h d -> b n s h d", s=s)
            k = rearrange(key, "b (n s) h d -> b n s h d", s=s)
            v = rearrange(value, "b (n s) h d -> b n s h d", s=s)
            # Apply dilation and segment offset
            offset = i % r
            hmin = i * group_size
            hmax = (i + 1) * group_size
            q = q[:, :, offset::r, hmin:hmax, :]
            k = k[:, :, offset::r, hmin:hmax, :]
            v = v[:, :, offset::r, hmin:hmax, :]
            # Fold all 'n' segments into the batch dimension
            q = rearrange(q, "b n s h d -> (b n) s h d")
            k = rearrange(k, "b n s h d -> (b n) s h d")
            v = rearrange(v, "b n s h d -> (b n) s h d")

            # Apply flash attention
            attn_bias = xops.LowerTriangularMask() if causal else None
            x = xops.memory_efficient_attention(
                query=q, key=k, value=v, op=self.op, attn_bias=attn_bias
            )
            # Unfold 'n' segments back out of the batch dimension.
            x = rearrange(x, "(b n) s h d -> b n s h d", b=b)

            # Gather the attention outputs from each dilation rate / segment length.
            out = rearrange(out, "b (n s) h d -> b n s h d", s=s)
            out[:, :, offset::r, hmin:hmax, :] += x
            out = rearrange(out, "b n s h d -> b (n s) h d", s=s)

        # Normalize attention outputs across the sequence length dimension.  This
        # is necessary because the attention outputs from each dilation rate /
        # segment length are summed together.
        # TODO: Double-check that we're summing over the correct dimension.
        return out / out.sum(dim=1, keepdim=True)


class MultiheadDilatedAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dilation_rates: Sequence[int],
        segment_lengths: Sequence[int],
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        op: Optional[xops.AttentionOp] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dilation_rates = dilation_rates
        self.segment_lengths = segment_lengths
        self.dropout = dropout
        self.bias = bias
        self.batch_first = batch_first

        if not batch_first:
            raise ValueError("batch_first=False is not supported")
        elif not self.embed_dim % self.num_heads == 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        num_dilations = len(dilation_rates)
        num_segments = len(segment_lengths)
        if num_dilations != num_segments:
            raise ValueError(
                f"len(dilation_rates) ({num_dilations}) must be equal to "
                f"len(segment_lengths) ({num_segments})"
            )
        elif not num_heads % num_dilations == 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"len(dilation_rates) ({num_dilations})"
            )
        self.head_dim = self.embed_dim // num_heads
        if not self.head_dim % 8 == 0:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {self.head_dim}) must be "
                "divisible by 8"
            )
        if not self.head_dim <= 128:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {self.head_dim}) must be <= 128"
            )

        self.w_q = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.w_k = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.w_v = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.attention = DilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            attention_dropout=dropout,
            op=op,
        )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )

    # TODO: Better initialization of weights and biases.
    # NOTE: This is an example pulled from 'nn.MultiheadAttention'.
    # def _reset_parameters(self):
    #     if self._qkv_same_embed_dim:
    #         xavier_uniform_(self.in_proj_weight)
    #     else:
    #         xavier_uniform_(self.q_proj_weight)
    #         xavier_uniform_(self.k_proj_weight)
    #         xavier_uniform_(self.v_proj_weight)

    #     if self.in_proj_bias is not None:
    #         constant_(self.in_proj_bias, 0.)
    #         constant_(self.out_proj.bias, 0.)
    #     if self.bias_k is not None:
    #         xavier_normal_(self.bias_k)
    #     if self.bias_v is not None:
    #         xavier_normal_(self.bias_v)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, is_causal: bool = False
    ) -> Tuple[Tensor, None]:
        # Notation:
        #   b - batch size
        #   n - sequence length
        #   h - number of heads
        #   d - embedding dimension
        #
        # Input shape: (b, n, d)
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        # Unfold 'd' dimension into 'h' separate attention heads.
        q = rearrange(q, "b n (h d) -> b n h d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b n h d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b n h d", h=self.num_heads)
        # Apply attention, then fold 'h' attention heads back into 'd'.
        x = self.attention(q, k, v, causal=is_causal)
        x = rearrange(x, "b n h d -> b n (h d)")

        # Linear projection on attention outputs.
        x = self.out_proj(x)

        return x, None


if __name__ == "__main__":
    torch.cuda.empty_cache()

    # x = torch.randn(4, 8192, 512, dtype=torch.float16, device="cuda")
    # mha = nn.MultiheadAttention(
    #     embed_dim=512,
    #     num_heads=8,
    #     device="cuda",
    #     dtype=torch.float16,
    #     batch_first=True,
    # )
    # mhda = MultiheadDilatedAttention(
    #     embed_dim=512,
    #     num_heads=8,
    #     segment_lengths=[64, 128, 256, 512],
    #     dilation_rates=[1, 2, 4, 8],
    #     device="cuda",
    #     dtype=torch.float16,
    #     op=xops.MemoryEfficientAttentionFlashAttentionOp,
    # )
    # with torch.no_grad():
    #     y1, _ = mha(x, x, x)
    #     y2, _ = mhda(x, x, x)

    x = torch.randn(8, 2**18, 8, 64, dtype=torch.float16, device="cuda")
    dilated_attention = DilatedAttention(
        segment_lengths=[64, 128, 256, 512],
        dilation_rates=[1, 2, 4, 8],
    )
    attention = xops.memory_efficient_attention

    with torch.no_grad():
        y1 = attention(x, x, x)
        y2 = dilated_attention(x, x, x)

    # from torch.profiler import profile, record_function, ProfilerActivity
    #
    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    #     for _ in range(100):
    #         with record_function("mhda"):
    #             y1, _ = mhda(x, x, x)
    #         # torch.cuda.synchronize()
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
