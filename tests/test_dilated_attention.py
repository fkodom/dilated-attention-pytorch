from typing import Tuple

import pytest
import torch

from dilated_attention_pytorch.dilated_attention import (
    DilatedAttention,
    MultiheadDilatedAttention,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
SEQ_LEN = 16


@pytest.mark.parametrize("segment_lengths", [(4,), (2, 4, 8), (2, 4)])
@pytest.mark.parametrize("dilation_rates", [(1,), (1, 2, 4), (1,)])
@pytest.mark.parametrize("embed_dim", [32])
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("is_causal", [True, False])
def test_dilated_attention(
    segment_lengths: Tuple[int, ...],
    dilation_rates: Tuple[int, ...],
    embed_dim: int,
    num_heads: int,
    is_causal: bool,
):
    if len(segment_lengths) != len(dilation_rates):
        with pytest.raises(ValueError):
            DilatedAttention(segment_lengths, dilation_rates)
        return

    dilated_attention = DilatedAttention(segment_lengths, dilation_rates)
    x = torch.randn(1, SEQ_LEN, num_heads, embed_dim, device=DEVICE, dtype=DTYPE)

    # Causal attention is not implemented on CPU for 'xformers'.
    if is_causal and DEVICE == torch.device("cpu"):
        with pytest.raises(NotImplementedError):
            dilated_attention(x, x, x, is_causal=is_causal)
        return

    out = dilated_attention(x, x, x, is_causal=is_causal)  # default: causal=False
    assert out.size(0) == 1
    assert out.size(1) == SEQ_LEN
    assert out.size(2) == num_heads
    assert out.size(3) == embed_dim


@torch.no_grad()
@pytest.mark.parametrize("segment_lengths", [(4,), (2, 4, 8), (2, 4)])
@pytest.mark.parametrize("dilation_rates", [(1,), (1, 2, 4), (1,)])
@pytest.mark.parametrize("embed_dim", [64, 128])
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("is_causal", [True, False])
def test_multihead_dilated_attention(
    segment_lengths: Tuple[int, ...],
    dilation_rates: Tuple[int, ...],
    embed_dim: int,
    num_heads: int,
    is_causal: bool,
):
    if len(segment_lengths) != len(dilation_rates):
        with pytest.raises(ValueError):
            MultiheadDilatedAttention(
                embed_dim, num_heads, segment_lengths, dilation_rates
            )
        return

    mhda = MultiheadDilatedAttention(
        embed_dim,
        num_heads,
        segment_lengths,
        dilation_rates,
        device=DEVICE,
        dtype=DTYPE,
    )
    x = torch.randn(1, SEQ_LEN, embed_dim, device=DEVICE, dtype=DTYPE)

    # Causal attention is not implemented on CPU for 'xformers'.
    if is_causal and DEVICE == torch.device("cpu"):
        with pytest.raises(NotImplementedError):
            mhda(x, x, x, is_causal=is_causal)
        return

    out, _ = mhda(x, x, x, is_causal=is_causal)  # default: causal=False
    assert out.size(0) == 1
    assert out.size(1) == SEQ_LEN
    assert out.size(2) == embed_dim
