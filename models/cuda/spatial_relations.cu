#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>

namespace {

constexpr int kNumPlayerFeatures = 4;
constexpr int kNumRelationFeatures = 6;

__global__ void spatial_relations_kernel(
    const float* __restrict__ players,
    float* __restrict__ output,
    int batch_size,
    int num_players
) {
    const int64_t total_pairs =
        static_cast<int64_t>(batch_size) *
        num_players *
        num_players;

    const int64_t index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= total_pairs) {
        return;
    }

    // Flattened layout:
    // index = ((b * N) + i) * N + j
    const int j = index % num_players;
    const int64_t temp = index / num_players;
    const int i = temp % num_players;
    const int b = temp / num_players;

    const int64_t i_offset =
        (static_cast<int64_t>(b) * num_players + i) * kNumPlayerFeatures;

    const int64_t j_offset =
        (static_cast<int64_t>(b) * num_players + j) * kNumPlayerFeatures;

    const float xi  = players[i_offset + 0];
    const float yi  = players[i_offset + 1];
    const float vxi = players[i_offset + 2];
    const float vyi = players[i_offset + 3];

    const float xj  = players[j_offset + 0];
    const float yj  = players[j_offset + 1];
    const float vxj = players[j_offset + 2];
    const float vyj = players[j_offset + 3];

    // Player j relative to player i.
    const float dx = xj - xi;
    const float dy = yj - yi;

    const float dvx = vxj - vxi;
    const float dvy = vyj - vyi;

    const float distance_sq = dx * dx + dy * dy;
    const float distance = sqrtf(distance_sq + 1e-8f);

    const float inv_distance = 1.0f / distance;
    const float unit_dx = dx * inv_distance;
    const float unit_dy = dy * inv_distance;

    // Positive when the players are moving toward one another.
    const float closing_speed =
        -(dvx * unit_dx + dvy * unit_dy);

    const int64_t out_offset = index * kNumRelationFeatures;

    output[out_offset + 0] = dx;
    output[out_offset + 1] = dy;
    output[out_offset + 2] = distance;
    output[out_offset + 3] = dvx;
    output[out_offset + 4] = dvy;
    output[out_offset + 5] = closing_speed;
}

}  // namespace


torch::Tensor spatial_relations_cuda(torch::Tensor players) {
    TORCH_CHECK(players.is_cuda(), "players must be a CUDA tensor");
    TORCH_CHECK(
        players.scalar_type() == torch::kFloat32,
        "players must have dtype float32"
    );
    TORCH_CHECK(players.dim() == 3, "players must have shape [B, N, 4]");
    TORCH_CHECK(
        players.size(2) == kNumPlayerFeatures,
        "last dimension must have size 4"
    );
    TORCH_CHECK(players.is_contiguous(), "players must be contiguous");

    const int batch_size = static_cast<int>(players.size(0));
    const int num_players = static_cast<int>(players.size(1));

    auto output = torch::empty(
        {batch_size, num_players, num_players, kNumRelationFeatures},
        players.options()
    );

    const int64_t total_pairs =
        static_cast<int64_t>(batch_size) *
        num_players *
        num_players;

    constexpr int threads = 256;

    const int blocks = static_cast<int>(
        (total_pairs + threads - 1) / threads
    );

    // Important: use PyTorch's CURRENT stream, not CUDA stream 0.
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    spatial_relations_kernel<<<blocks, threads, 0, stream>>>(
        players.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_players
    );

    const cudaError_t error = cudaGetLastError();

    TORCH_CHECK(
        error == cudaSuccess,
        "spatial_relations_kernel launch failed: ",
        cudaGetErrorString(error)
    );

    return output;
}