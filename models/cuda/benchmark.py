from __future__ import annotations

import time

import torch

import pivot_cuda
from pivot.model import pairwise_relations_torch


NUM_PLAYERS = 22
PLAYER_FEATURES = 4


def cuda_relations(players: torch.Tensor) -> torch.Tensor:
    return pivot_cuda.spatial_relations(players.contiguous())


@torch.no_grad()
def check_correctness(batch_size: int = 64) -> None:
    players = torch.randn(
        batch_size,
        NUM_PLAYERS,
        PLAYER_FEATURES,
        device="cuda",
        dtype=torch.float32,
    )

    expected = pairwise_relations_torch(players)
    actual = cuda_relations(players)

    max_error = (expected - actual).abs().max().item()
    mean_error = (expected - actual).abs().mean().item()

    print(f"Correctness test, batch={batch_size}")
    print(f"  max error : {max_error:.8e}")
    print(f"  mean error: {mean_error:.8e}")

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-5,
    )

    print("  PASS")


@torch.no_grad()
def benchmark_function(
    fn,
    players: torch.Tensor,
    warmup: int = 100,
    iterations: int = 1000,
) -> float:
    for _ in range(warmup):
        fn(players)

    torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(iterations):
        fn(players)

    torch.cuda.synchronize()

    end = time.perf_counter()

    return (end - start) * 1_000_000 / iterations


@torch.no_grad()
def run_benchmarks() -> None:
    print()
    print("RTX CUDA relation benchmark")
    print("=" * 60)

    for batch_size in [1, 32, 256, 1024, 4096]:
        players = torch.randn(
            batch_size,
            NUM_PLAYERS,
            PLAYER_FEATURES,
            device="cuda",
            dtype=torch.float32,
        )

        torch_us = benchmark_function(
            pairwise_relations_torch,
            players,
        )

        cuda_us = benchmark_function(
            cuda_relations,
            players,
        )

        speedup = torch_us / cuda_us

        print(
            f"B={batch_size:4d} | "
            f"PyTorch={torch_us:9.2f} us | "
            f"CUDA={cuda_us:9.2f} us | "
            f"speedup={speedup:6.2f}x"
        )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

    print()

    check_correctness()
    run_benchmarks()


if __name__ == "__main__":
    main()