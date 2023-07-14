import logging
from functools import partial
from math import ceil
from timeit import Timer
from typing import Callable, List, NamedTuple

import torch

import xformers.ops as xops
from dilated_attention_pytorch.dilated_attention import DilatedAttention

# Generic benchmarking parameters
BATCH_SIZE = 1
TOTAL_TOKENS = 2**25  # 32M
NUM_HEADS = 4
EMBED_DIM = 8
# Vanilla attention only
VANILLA_SEQ_LENGTHS = [2**i for i in range(13, 18)]  # 8k - 128k
# Dilated attention only
SEGMENT_LENGTHS = [8192, 16384, 32768]  # 8k - 64k
DILATED_SEQ_LENGTHS = [2**i for i in range(13, 26)]  # 8k - 32M


class BenchmarkResult(NamedTuple):
    mean: float
    std: float

    def __repr__(self):
        return f"BenchmarkResult(mean: {self.mean:.3e}, std: {self.std:.3e})"

    def __str__(self):
        return f"({self.mean:.3e} \u00B1 {self.std:.3e}) s"


def benchmark(
    fn: Callable,
    *args,
    min_total_seconds: float = 1.0,
    min_iterations: int = 2,
    **kwargs,
) -> BenchmarkResult:
    # Benchmark the runtime of a function and dynamically determine the number of
    # iterations to run.  Continue running the function until *total* runtime
    # exceeds 'min_total_seconds' and 'min_iterations'.
    if min_iterations < 2:
        raise ValueError("min_iterations must be >= 2")

    timer = Timer(
        "fn(*args, **kwargs)",
        globals={"fn": fn, "args": args, "kwargs": kwargs},
    )
    # Run the function once to warm up
    _ = timer.repeat(number=1, repeat=1)

    times: List[float] = []
    total_time = 0.0
    num_iterations = min_iterations or 1

    while total_time < min_total_seconds:
        _times = timer.repeat(number=1, repeat=num_iterations)
        times.extend(_times)
        _total_time = sum(_times)
        total_time += _total_time

        # Estimate how many more iterations we need to run to get to 1 second
        avg_time = _total_time / num_iterations
        num_iterations = ceil((min_total_seconds - total_time) / avg_time)

    times_tensor = torch.as_tensor(times)
    return BenchmarkResult(
        mean=times_tensor.mean().item(),
        std=times_tensor.std().item(),
    )


def get_dilated_attention_for_seq_length(seq_length: int) -> DilatedAttention:
    """This is roughly how benchmarking was described in the paper, except that they
    were testing in a distributed (multi-GPU) setting.  We use a base segment
    length of 8192, and include larger segment lengths if possible.  I believe
    this is the equivalent benchmark for 1 GPU.

    Reference:
        https://arxiv.org/pdf/2307.02486.pdf, Section 3.1
    """
    segment_lengths: list[int] = []
    dilation_rates: list[int] = []

    for segment_length in SEGMENT_LENGTHS:
        # We can't use segment lengths larger than the sequence length.
        segment_length = min(segment_length, seq_length)
        exponent = segment_length // SEGMENT_LENGTHS[0] - 1
        dilation_rate = 2**exponent

        segment_lengths.append(segment_length)
        dilation_rates.append(dilation_rate)

    return DilatedAttention(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        op=xops.MemoryEfficientAttentionFlashAttentionOp,
    )


def attention_forward(x: torch.Tensor, attn: Callable):
    with torch.no_grad():
        _ = attn(x, x, x)
    torch.cuda.synchronize()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logging.info("Benchmark vanilla attention...")
    vanilla_results: list[BenchmarkResult] = []
    for seq_length in VANILLA_SEQ_LENGTHS:
        torch.cuda.empty_cache()
        batch_size = TOTAL_TOKENS // seq_length
        x = torch.randn(
            (batch_size, seq_length, NUM_HEADS, EMBED_DIM),
            dtype=torch.float16,
            device="cuda",
        )
        fn = partial(attention_forward, attn=xops.memory_efficient_attention)
        result = benchmark(fn, x)
        vanilla_results.append(result)
        logging.info(f"Sequence length {seq_length}: {result}")

    logging.info("Benchmark dilated attention...")
    dilated_results: list[BenchmarkResult] = []
    for seq_length in DILATED_SEQ_LENGTHS:
        torch.cuda.empty_cache()
        batch_size = TOTAL_TOKENS // seq_length
        x = torch.randn(
            (batch_size, seq_length, NUM_HEADS, EMBED_DIM),
            dtype=torch.float16,
            device="cuda",
        )
        attn = get_dilated_attention_for_seq_length(seq_length)
        fn = partial(attention_forward, attn=attn)
        result = benchmark(fn, x)
        dilated_results.append(result)
        logging.info(f"Sequence length {seq_length}: {result}")
