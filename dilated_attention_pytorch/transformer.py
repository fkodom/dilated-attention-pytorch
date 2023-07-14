from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.transformer import _get_activation_fn

from dilated_attention_pytorch.dilated_attention import MultiheadDilatedAttention


class DilatedTransformerEncoderLayer(nn.Module):
    # NOTE: Mostly pulled from 'nn.TransformerEncoderLayer', but with changes:
    #   - use sub-LayerNorm like in MAGNETO. See: https://arxiv.org/abs/2210.06423
    #   - use MultiheadDilatedAttention instead of MultiheadAttention

    def __init__(
        self,
        d_model: int,
        nhead: int,
        segment_lengths: Sequence[int],
        dilation_rates: Sequence[int],
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        gamma_init: float = 1.0,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        self.activation = activation
        self.gamma_init = gamma_init

        self.dropout = nn.Dropout(dropout)
        # Self-attention block
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.self_attn = MultiheadDilatedAttention(  # type: ignore
            embed_dim=d_model,
            num_heads=nhead,
            dilation_rates=dilation_rates,
            segment_lengths=segment_lengths,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            gamma_init=gamma_init,
            device=device,
            dtype=dtype,
        )
        # Feedforward block
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, device=device, dtype=dtype)
        self.norm3 = nn.LayerNorm(
            dim_feedforward, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.linear2 = nn.Linear(dim_feedforward, d_model, device=device, dtype=dtype)

        self._reset_parameters()

    def _reset_parameters(self):
        # NOTE: We follow the initialization strategy from MAGNETO.  See:
        # https://arxiv.org/pdf/2210.06423.pdf, Fig. 2
        # The 'MultiheadDilatedAttention' module uses ths same initialization,
        # so we just need to worry about the 'Linear' modules here.
        nn.init.xavier_normal_(self.linear1.weight, gain=self.gamma_init)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.xavier_normal_(self.linear2.weight, gain=self.gamma_init)
        nn.init.constant_(self.linear2.bias, 0)

    def forward(self, src: Tensor, is_causal: bool = False) -> Tensor:
        x = src

        # Self-attention block
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, is_causal=is_causal)
        x = self.dropout(x)

        # Feedforward block
        x = self.norm2(x)
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.norm3(x)
        x = self.linear2(x)
        x = self.dropout(x)

        return x


class DilatedTransformerDecoderLayer(nn.Module):
    # NOTE: Mostly pulled from 'nn.TransformerDecoderLayer', but with changes:
    #   - use sub-LayerNorm like in MAGNETO. See: https://arxiv.org/abs/2210.06423
    #   - use MultiheadDilatedAttention instead of MultiheadAttention

    def __init__(
        self,
        d_model: int,
        nhead: int,
        segment_lengths: Sequence[int],
        dilation_rates: Sequence[int],
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        gamma_init: float = 1.0,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        self.activation = activation
        self.gamma_init = gamma_init

        self.dropout = nn.Dropout(dropout)
        # Self-attention block
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.self_attn = MultiheadDilatedAttention(  # type: ignore
            embed_dim=d_model,
            num_heads=nhead,
            dilation_rates=dilation_rates,
            segment_lengths=segment_lengths,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            gamma_init=gamma_init,
            device=device,
            dtype=dtype,
        )
        # Multi-head attention block
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.multihead_attn = MultiheadDilatedAttention(  # type: ignore
            embed_dim=d_model,
            num_heads=nhead,
            dilation_rates=dilation_rates,
            segment_lengths=segment_lengths,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            gamma_init=gamma_init,
            device=device,
            dtype=dtype,
        )
        # Feedforward block
        self.norm3 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, device=device, dtype=dtype)
        self.norm4 = nn.LayerNorm(
            dim_feedforward, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.linear2 = nn.Linear(dim_feedforward, d_model, device=device, dtype=dtype)

        self._reset_parameters()

    def _reset_parameters(self):
        # NOTE: We follow the initialization strategy from MAGNETO.  See:
        # https://arxiv.org/pdf/2210.06423.pdf, Fig. 2
        # The 'MultiheadDilatedAttention' module uses ths same initialization,
        # so we just need to worry about the 'Linear' modules here.
        nn.init.xavier_normal_(self.linear1.weight, gain=self.gamma_init)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.xavier_normal_(self.linear2.weight, gain=self.gamma_init)
        nn.init.constant_(self.linear2.bias, 0)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        x = tgt

        # Self-attention block
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, is_causal=tgt_is_causal)
        x = self.dropout(x)

        # Multihead attention block
        x = self.norm2(x)
        x, _ = self.multihead_attn(x, memory, memory, is_causal=memory_is_causal)
        x = self.dropout(x)

        # Feedforward block
        x = self.norm3(x)
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.norm4(x)
        x = self.linear2(x)
        x = self.dropout(x)

        return x
