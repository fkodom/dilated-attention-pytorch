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
        if len(segment_lengths) != len(dilation_rates):
            raise ValueError(
                "segment_lengths and dilation_rates must have the same length"
            )

        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.op = op

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, is_causal: bool = False
    ) -> Tensor:
        # Notation:
        #   b - batch size
        #   n - sequence length
        #   h - number of heads
        #   d - embedding dimension
        #   s - segment length
        #   r - dilation rate
        #   g - group size (i.e. number of heads per segment length)
        #
        # Input shape of query, key, value: (b, n, h, d)
        b, _, h, _ = query.shape
        out = torch.zeros_like(query)

        # *** NOTE ***
        # The original paper does not describe how to handle the case where
        #   h % len(self.segment_lengths) != 0
        #
        # In my first implementation, I naively assumed (and asserted) that
        # 'h % len(self.segment_lengths) == 0', so that I could evenly distribute
        # the heads between the different segment lengths. However, it was not
        # possible to reproduce the LongNet hyperparameters with that restriction:
        #   h=12, segment_lengths=[2048, 4096, 8192, 16384, 32768]
        #   h % len(segment_lengths) == 2
        #
        # For that reason, I have removed the assertion, and instead grouped the heads
        # into (potentially) unequally sized groups.  If not perfectly divisible, then
        # the first few groups will have an extraattention head.
        num_groups = len(self.dilation_rates)
        group_sizes = [h // num_groups] * num_groups
        for i in range(h % num_groups):
            group_sizes[i] += 1

        for i, (g, r, s) in enumerate(
            zip(group_sizes, self.dilation_rates, self.segment_lengths)
        ):
            # Split the input sequences into segments of length 'self.segment_length'
            q = rearrange(query, "b (n s) h d -> b n s h d", s=s)
            k = rearrange(key, "b (n s) h d -> b n s h d", s=s)
            v = rearrange(value, "b (n s) h d -> b n s h d", s=s)
            # Apply dilation and segment offset
            offset = i % r
            hmin = i * g
            hmax = (i + 1) * g
            q = q[:, :, offset::r, hmin:hmax, :]
            k = k[:, :, offset::r, hmin:hmax, :]
            v = v[:, :, offset::r, hmin:hmax, :]
            # Fold all 'n' segments into the batch dimension
            q = rearrange(q, "b n s h d -> (b n) s h d")
            k = rearrange(k, "b n s h d -> (b n) s h d")
            v = rearrange(v, "b n s h d -> (b n) s h d")

            # Apply memory efficient attention
            # NOTE: If flash attention is correctly installed, then this will also
            # automatically use the flash attention implementation.
            attn_bias = xops.LowerTriangularMask() if is_causal else None
            x = xops.memory_efficient_attention(
                query=q, key=k, value=v, op=self.op, attn_bias=attn_bias
            )
            # Unfold 'n' segments back out of the batch dimension.
            x = rearrange(x, "(b n) s h d -> b n s h d", b=b)
            # Normalize attention outputs across the sequence length dimension. This
            # is necessary because the attention outputs from each dilation rate /
            # segment length are summed together.
            x = x / x.sum(dim=(1, 2), keepdim=True)

            # Gather the attention outputs from each dilation rate / segment length.
            out = rearrange(out, "b (n s) h d -> b n s h d", s=s)
            out[:, :, offset::r, hmin:hmax, :] += x
            out = rearrange(out, "b n s h d -> b (n s) h d", s=s)

        # We have already normalized each attention output across the sequence length.
        # Now, normalize across all attention outputs by dividing by the number of
        # attention groups.  See: https://arxiv.org/pdf/2307.02486.pdf, Eq. 10
        return out / num_groups


class MultiheadDilatedAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dilation_rates: Sequence[int],
        segment_lengths: Sequence[int],
        dropout: float = 0.0,
        bias: bool = True,
        layer_norm: bool = True,
        layer_norm_eps: float = 1e-5,
        gamma_init: float = 1.0,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        op: Optional[xops.AttentionOp] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.gamma_init = gamma_init

        if not embed_dim % self.num_heads == 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        num_dilations = len(dilation_rates)
        num_segments = len(segment_lengths)
        if num_dilations != num_segments:
            raise ValueError(
                f"len(dilation_rates) ({num_dilations}) must be equal to "
                f"len(segment_lengths) ({num_segments})"
            )
        head_dim = embed_dim // num_heads
        if not head_dim % 8 == 0:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {head_dim}) must be divisible by 8"
            )
        if not head_dim <= 128:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {head_dim}) must be <= 128"
            )

        self.q_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.k_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.v_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.attention = DilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            attention_dropout=dropout,
            op=op,
        )
        self.norm: Optional[nn.LayerNorm] = None
        if layer_norm:
            self.norm = nn.LayerNorm(
                embed_dim, eps=layer_norm_eps, device=device, dtype=dtype
            )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
        nn.init.xavier_normal_(self.k_proj.weight)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0)

        # NOTE: We follow the initialization strategy from MAGNETO.  See:
        # https://arxiv.org/pdf/2210.06423.pdf, Fig. 2
        # Gain (self.gamma_init) should be provided as a keyword argument when
        # initializing the larger Transformer model, since it requires knowledge
        # of the number of encoder/decoder layers in the model.

        nn.init.xavier_normal_(self.v_proj.weight, gain=self.gamma_init)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0)
        nn.init.xavier_normal_(self.out_proj.weight, gain=self.gamma_init)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

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
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Unfold 'd' dimension into 'h' separate attention heads.
        q = rearrange(q, "b n (h d) -> b n h d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b n h d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b n h d", h=self.num_heads)
        # Apply attention, then fold 'h' attention heads back into 'd'.
        x = self.attention(q, k, v, is_causal=is_causal)
        x = rearrange(x, "b n h d -> b n (h d)")

        # NOTE: This is different from 'nn.MultiheadAttention'! The LongNet paper
        # follows the MAGNETO architecture, which applies an extra layer norm
        # before the linear output projection.  The cross-attention layer in the
        # MAGNETO decoder does not include this layer norm, so users have the option
        # to disable it (layer_norm=False).
        if self.layer_norm:
            assert self.norm is not None
            x = self.norm(x)
        # Linear projection on attention outputs.
        x = self.out_proj(x)

        return x, None
