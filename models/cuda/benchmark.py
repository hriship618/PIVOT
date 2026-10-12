"""Correctness and speed benchmark for the PIVOT CUDA relation kernel."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch


MODELS_DIR = Path(
    __file__
).resolve().parents[1]

if str(MODELS_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(MODELS_DIR),
    )


import pivot_cuda

from pivot.model import (
    pairwise_relations,
)


def cuda_relations(
    players: torch.Tensor,
) -> torch.Tensor:
    return pivot_cuda.spatial_relations(
        players
    )


def benchmark(
    function,
    players: torch.Tensor,
    iterations: int = 1000,
) -> float:
    """Return average kernel/runtime latency in milliseconds."""

    # Warm up kernels, allocations and CUDA runtime.
    for _ in range(100):
        function(players)

    torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(iterations):
        function(players)

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    return (
        elapsed
        / iterations
        * 1000.0
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available"
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    torch.manual_seed(42)

    batch_sizes = [
        1,
        32,
        256,
        1024,
        4096,
    ]

    for batch_size in batch_sizes:
        players = torch.randn(
            batch_size,
            22,
            4,
            device="cuda",
            dtype=torch.float32,
        )

        reference = pairwise_relations(
            players
        )

        custom = cuda_relations(
            players
        )

        max_error = (
            reference
            - custom
        ).abs().max().item()

        if max_error > 1e-5:
            raise RuntimeError(
                f"CUDA result mismatch: max error={max_error}"
            )

        torch_ms = benchmark(
            pairwise_relations,
            players,
        )

        cuda_ms = benchmark(
            cuda_relations,
            players,
        )

        speedup = (
            torch_ms / cuda_ms
            if cuda_ms > 0
            else float("inf")
        )

        print(
            f"B={batch_size:4d} | "
            f"PyTorch={torch_ms:.4f} ms | "
            f"CUDA={cuda_ms:.4f} ms | "
            f"speedup={speedup:.2f}x | "
            f"max_error={max_error:.2e}"
        )


if __name__ == "__main__":
    main()