from typing import Callable, Tuple, Union

import pytest
import torch

from dilated_attention_pytorch.long_net import LongNet, LongNetLM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
SEQ_LEN = 16


@pytest.fixture(params=[(4,), (4, 8, 16)])
def segment_lengths(request) -> Tuple[int, ...]:
    return request.param


@pytest.fixture()
def dilation_rates(segment_lengths: Tuple[int, ...]):
    return tuple([s // 4 for s in segment_lengths])


@pytest.mark.parametrize("d_model", [128])
@pytest.mark.parametrize("nhead", [4, 8])
@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("dim_feedforward", [64])
@pytest.mark.parametrize("dropout", [0.0])
@pytest.mark.parametrize("activation", ["relu", "gelu"])
# NOTE: 'is_causal=True' causes issues on CPU
@pytest.mark.parametrize("is_causal", [False])
def test_long_net(
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    segment_lengths: Tuple[int, ...],
    dilation_rates: Tuple[int, ...],
    dropout: float,
    activation: Union[str, Callable],
    is_causal: bool,
):
    net = LongNet(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        dim_feedforward=dim_feedforward,
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        dropout=dropout,
        activation=activation,
        device=DEVICE,
        dtype=DTYPE,
    )
    x = torch.randn(1, SEQ_LEN, d_model, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        y = net.forward(x, is_causal=is_causal)
    assert y.size(0) == 1
    assert y.size(1) == SEQ_LEN
    assert y.size(2) == d_model


@pytest.mark.parametrize("num_tokens", [100])
@pytest.mark.parametrize("d_model", [128])
@pytest.mark.parametrize("nhead", [4, 8])
@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("dim_feedforward", [64])
@pytest.mark.parametrize("dropout", [0.0])
@pytest.mark.parametrize("activation", ["relu", "gelu"])
# NOTE: 'is_causal=True' causes issues on CPU
@pytest.mark.parametrize("is_causal", [False])
def test_long_net_lm(
    num_tokens: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    segment_lengths: Tuple[int, ...],
    dilation_rates: Tuple[int, ...],
    dropout: float,
    activation: Union[str, Callable],
    is_causal: bool,
):
    net = LongNetLM(
        num_tokens=num_tokens,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        dim_feedforward=dim_feedforward,
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        dropout=dropout,
        activation=activation,
        device=DEVICE,
        dtype=DTYPE,
    )
    x = torch.randint(0, num_tokens, (1, SEQ_LEN), device=DEVICE, dtype=torch.long)
    with torch.no_grad():
        y = net.forward(x, is_causal=is_causal)
    assert y.size(0) == 1
    assert y.size(1) == SEQ_LEN
    assert y.size(2) == num_tokens
