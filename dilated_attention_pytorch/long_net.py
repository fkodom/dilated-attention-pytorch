from math import log
from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchscale.component.xpos_relative_position import XPOS

from dilated_attention_pytorch.transformer import (
    DilatedTransformerDecoderLayer,
    DilatedTransformerEncoderLayer,
)


class LongNet(nn.Module):
    """These are the *base* LongNet hyperparameters taken from the paper.  See:
    https://arxiv.org/pdf/2307.02486.pdf, Section 4.1 & Appendix A

    NOTE - Differences from the paper:
    - 'segment_lengths' is [2048, 4096, 8192, 16384, 32768, 32768]
                instead of [2048, 4096, 8192, 16384, 32768]
    - 'dilation_rates' is [1, 2, 4, 6, 12, 12]
               instead of [1, 2, 4, 6, 12]

    Explanation:
    - 'd_model' must be divisible by 'nheads', so that we can split the embedding
      dimension evenly across heads. This is not uncommon for self-attention layers.
    - My implementation of 'DilatedAttention' also requires that 'nheads' is divisible
      by 'len(dilation_rates)', so that we can split the heads evenly across each
      dilation rate.  NOTE: THIS IS NOT MENTIONED AS A REQUIREMENT IN THE PAPER.
      I will continue looking into it.  :)
    - I didn't want to use any sequence lengths less than 2048 or greater than 32768,
      since those are the min/max used in the paper.  (Changing those would effectively
      change the attention window size, which I do not want to do.)  The simplest
      short-term solution was to add another (identical) segment length and dilation.
      We have 2x more attention heads at segment_length=32768 than we do for any other
      segment length, which means we spend more computation on long-range dependencies.
    """

    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 12,
        num_encoder_layers: int = 12,
        num_decoder_layers: int = 12,
        dim_feedforward: int = 3072,
        segment_lengths: Sequence[int] = [2048, 4096, 8192, 16384, 32768, 32768],
        dilation_rates: Sequence[int] = [1, 2, 4, 6, 12, 12],
        dropout: float = 0.0,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        # The 'gamma_init' parameters are different for the encoder and decoder,
        # and depend on the number of encoder/decoder layers.  See MAGNETO paper:
        # https://arxiv.org/pdf/2210.06423.pdf, Figure 2
        encoder_gamma_init = (
            log(3 * num_decoder_layers) * log(2 * num_encoder_layers) / 3
        ) ** 0.5
        decoder_gamma_init = log(3 * num_decoder_layers) ** 0.5

        self.encoder = nn.TransformerEncoder(
            encoder_layer=DilatedTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                gamma_init=encoder_gamma_init,
                device=device,
                dtype=dtype,
            ),
            num_layers=num_encoder_layers,
            mask_check=False,
            enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=DilatedTransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                gamma_init=decoder_gamma_init,
                device=device,
                dtype=dtype,
            ),
            num_layers=num_decoder_layers,
        )

    def forward(self, x: Tensor, is_causal: bool = True) -> Tensor:
        """
        Input shape: (batch_size, seq_len, d_model)
        Output shape: (batch_size, seq_len, d_model)

        NOTE: Assume that 'is_causal' applies to both the encoder and decoder.
        We're primarily interested in causal attention for language modeling, which is
        what was discussed in the LongNet paper.  But in principle, leave the option
        open for other applications.
        """
        tgt = x
        for layer in self.encoder.layers:
            x = layer(x, is_causal=is_causal)
        if self.encoder.norm is not None:
            x = self.encoder.norm(x)

        mem = x
        for layer in self.decoder.layers:
            tgt = layer(tgt, mem, memory_is_causal=is_causal, tgt_is_causal=is_causal)
        if self.decoder.norm is not None:
            tgt = self.decoder.norm(tgt)

        return tgt


class LongNetLM(nn.Module):
    """
    NOTE: There are some hyperparameter differences from the original paper.
    See 'LongNet' class for more details.
    """

    def __init__(
        self,
        num_tokens: int,
        d_model: int = 768,
        nhead: int = 12,
        num_encoder_layers: int = 12,
        num_decoder_layers: int = 12,
        dim_feedforward: int = 3072,
        segment_lengths: Sequence[int] = [2048, 4096, 8192, 16384, 32768, 32768],
        dilation_rates: Sequence[int] = [1, 2, 4, 6, 12, 12],
        dropout: float = 0.0,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(
            num_tokens, d_model, device=device, dtype=dtype
        )
        self.pos_embedding = XPOS(d_model).to(device=device, dtype=dtype)
        self.long_net = LongNet(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.norm = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.out = nn.Linear(d_model, num_tokens, device=device, dtype=dtype)

    def _reset_parameters(self):
        nn.init.kaiming_normal_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)

    def forward(self, x: Tensor, is_causal: bool = True) -> Tensor:
        x = self.token_embedding(x)
        x = x + self.pos_embedding(x)
        x = self.long_net(x, is_causal=is_causal)
        x = self.norm(x)
        return self.out(x)


if __name__ == "__main__":
    num_tokens = 1024
    device = torch.device("cuda")
    dtype = torch.float16

    x = torch.randint(0, num_tokens - 1, size=(2, 32768), device=device)
    model = LongNetLM(num_tokens=num_tokens, device=device, dtype=dtype)

    with torch.no_grad():
        out = model(x)

    print(out.shape)
    breakpoint()
    pass
