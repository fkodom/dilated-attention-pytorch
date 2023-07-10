from typing import Optional, Sequence, Tuple

import torch
from einops import rearrange
from flash_attn.flash_attention import FlashAttention
from torch import Tensor, nn


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
        dilation_rates: Sequence[int],
        sequence_lengths: Sequence[int],
        softmax_scale: Optional[float] = None,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.dilation_rates = dilation_rates
        self.sequence_lengths = sequence_lengths
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.attention = FlashAttention(
            softmax_scale=softmax_scale, attention_dropout=attention_dropout
        )

    def forward(self, qkv: Tensor, causal: bool = False):
        # Notation:
        #   b - batch size
        #   n - sequence length
        #   qkv - query, key, value
        #   h - number of heads
        #   d - embedding dimension
        #   r - dilation rate
        #   s - segment length
        #
        # Input shape of qkv: (b, n, 3, h, d)
        b, n, _, h, d = qkv.shape
        out = torch.zeros((b, n, h, d), dtype=qkv.dtype, device=qkv.device)

        for r, s in zip(self.dilation_rates, self.sequence_lengths):
            # Split the input sequences into segments of length 'self.segment_length'.
            # Then, apply dilation along the segment dimension, and fold the segments
            # into the batch dimension.
            x = rearrange(qkv, "b (n s) qkv h d -> b n s qkv h d", s=s)
            # Apply dilation
            x = x[:, :, ::r, :, :]
            # Fold segments into batch dimension
            x = rearrange(x, "b n s qkv h d -> (b s) n qkv h d")
            # Apply flash attention
            x, _ = self.attention(x, causal=causal)  # shape: (b * s, n, h, d)
            # Unfold segments back from the sequence dimension.
            x = rearrange(x, "(b s) n h d -> b n s h d", b=b)

            # Sum the attention outputs from each dilation rate / segment length.
            # NOTE: After dilation, 'x' has shape: (b, n // r, 3, h, d)
            out = rearrange(out, "b (n s) h d -> b n s h d", s=s)
            out[:, :, ::r, :, :] += x
            out = rearrange(out, "b n s h d -> b (n s) h d", s=s)
            print(qkv.shape, out.shape)

        # Normalize attention outputs across the sequence length dimension.  This
        # is necessary because the attention outputs from each dilation rate /
        # segment length are summed together.
        # TODO: Double-check that we're summing over the correct dimension.
        return out / out.sum(dim=1, keepdim=True), None


class DilatedMHA(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dilation_rates: Sequence[int],
        segment_lengths: Sequence[int],
        attention_dropout: float = 0.0,
        causal: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dilation_rates = dilation_rates
        self.segment_lengths = segment_lengths
        self.dropout = attention_dropout
        self.causal = causal
        self.bias = bias

        if not self.embed_dim % self.num_heads == 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        self.head_dim = self.embed_dim // num_heads
        if not self.head_dim % 8 == 0:
            raise ValueError(f"embed_dim ({self.head_dim}) must be divisible by 8")
        if not self.head_dim <= 128:
            raise ValueError(f"embed_dim ({self.head_dim}) must be <= 128")

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.inner_attn = DilatedAttention(
            dilation_rates=dilation_rates,
            sequence_lengths=segment_lengths,
            attention_dropout=attention_dropout,
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # Notation:
        #   b - batch size
        #   n - sequence length
        #   qkv - query, key, value
        #   h - number of heads
        #   d - embedding dimension
        #
        # Input shape of x: (b, n, d_model)
        qkv = self.Wqkv(x)  # shape: (b, n, 3 * h * d)
        qkv = rearrange(qkv, "b n (qkv h d) -> b n qkv h d", qkv=3, h=self.num_heads)
        x, attn_weights = self.inner_attn(qkv, causal=self.causal)
        x = rearrange(x, "b n h d -> b n (h d)")
        return self.out_proj(x), attn_weights


class DilatedTransformerEncoderLayer(nn.TransformerEncoderLayer):
    # TODO: Rewrite this to be independent of nn.TransformerEncoderLayer.
    # Out of laziness, I quickly generated this class using Copilot, based on
    # the source code of nn.TransformerEncoderLayer.  But it's inefficient, because
    # it effectly allocates the self-attention and linear layers twice.  :(
    #
    # I'm tired and don't want to deal with it right now.  :P

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dilation_rates: Sequence[int],
        segment_lengths: Sequence[int],
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.self_attn = DilatedMHA(  # type: ignore
            embed_dim=d_model,
            num_heads=nhead,
            dilation_rates=dilation_rates,
            segment_lengths=segment_lengths,
            attention_dropout=dropout,
            causal=False,
            bias=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


if __name__ == "__main__":
    x = torch.randn(8, 128, 512, dtype=torch.float16, device="cuda")
    mha = DilatedMHA(
        embed_dim=512,
        num_heads=8,
        segment_lengths=[256, 512, 1024],
        dilation_rates=[1, 2, 4],
    ).to(dtype=torch.float16, device="cuda")
    y, _ = mha(x)

    print(x.shape, y.shape)
    breakpoint()
    pass
