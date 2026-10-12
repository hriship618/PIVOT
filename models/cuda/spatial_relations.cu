#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>


// Each CUDA thread handles one directed player pair:
//
//     (batch, player_i, player_j)
//
// and writes the six relationship features:
//
//     dx
//     dy
//     distance
//     relative vx
//     relative vy
//     closing speed
//
// Input:
//     players [B, N, 4]
//
// Output:
//     relations [B, N, N, 6]
//
// Player features are:
//
//     x, y, vx, vy

__global__ void spatial_relations_kernel(
    const float* __restrict__ players,
    float* __restrict__ output,
    int batch_size,
    int num_players
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int pairs_per_batch = num_players * num_players;
    int total_pairs = batch_size * pairs_per_batch;

    if (index >= total_pairs) {
        return;
    }

    // Convert flat thread index into:
    //
    //     batch
    //     source player i
    //     destination player j

    int batch = index / pairs_per_batch;
    int pair_index = index % pairs_per_batch;

    int i = pair_index / num_players;
    int j = pair_index % num_players;


    // players is laid out:
    //
    // [batch][player][feature]
    //
    // with four features per player.

    int i_offset = (
        batch * num_players * 4
        + i * 4
    );

    int j_offset = (
        batch * num_players * 4
        + j * 4
    );


    float xi = players[i_offset + 0];
    float yi = players[i_offset + 1];
    float vxi = players[i_offset + 2];
    float vyi = players[i_offset + 3];

    float xj = players[j_offset + 0];
    float yj = players[j_offset + 1];
    float vxj = players[j_offset + 2];
    float vyj = players[j_offset + 3];


    // Relation i -> j means:
    //
    // where is j relative to i?

    float dx = xj - xi;
    float dy = yj - yi;

    float dvx = vxj - vxi;
    float dvy = vyj - vyi;


    // Small epsilon matches the PyTorch reference implementation and avoids
    // division by zero for self-edges i == j.

    float distance = sqrtf(
        dx * dx
        + dy * dy
        + 1e-8f
    );


    float unit_dx = dx / distance;
    float unit_dy = dy / distance;


    // Positive closing speed means the players are moving toward each other.
    //
    // Relative velocity projected onto the line joining the players:
    //
    //     dv dot unit_direction
    //
    // is negative when separation is decreasing, hence the minus sign.

    float closing_speed = -(
        dvx * unit_dx
        + dvy * unit_dy
    );


    // Output layout:
    //
    // [B, N, N, 6]

    int output_offset = (
        (
            batch * num_players * num_players
            + i * num_players
            + j
        )
        * 6
    );


    output[output_offset + 0] = dx;
    output[output_offset + 1] = dy;
    output[output_offset + 2] = distance;
    output[output_offset + 3] = dvx;
    output[output_offset + 4] = dvy;
    output[output_offset + 5] = closing_speed;
}


torch::Tensor spatial_relations_cuda(
    torch::Tensor players
) {
    TORCH_CHECK(
        players.is_cuda(),
        "players must be a CUDA tensor"
    );

    TORCH_CHECK(
        players.scalar_type() == torch::kFloat32,
        "players must currently be float32"
    );

    TORCH_CHECK(
        players.dim() == 3,
        "players must have shape [B, N, 4]"
    );

    TORCH_CHECK(
        players.size(2) == 4,
        "last player dimension must be 4"
    );

    TORCH_CHECK(
        players.is_contiguous(),
        "players must be contiguous"
    );


    int batch_size = players.size(0);
    int num_players = players.size(1);


    auto output = torch::empty(
        {
            batch_size,
            num_players,
            num_players,
            6
        },
        players.options()
    );


    int total_pairs = (
        batch_size
        * num_players
        * num_players
    );


    // 256 threads is a conventional starting point. CUDA divides the pair
    // computation across enough blocks to cover the complete batch.

    const int threads = 256;

    const int blocks = (
        total_pairs
        + threads
        - 1
    ) / threads;


    spatial_relations_kernel<<<
        blocks,
        threads
    >>>(
        players.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_players
    );


    // Surface kernel launch errors immediately instead of allowing them to
    // appear later in an unrelated PyTorch operation.

    cudaError_t error = cudaGetLastError();

    TORCH_CHECK(
        error == cudaSuccess,
        "spatial_relations CUDA kernel failed: ",
        cudaGetErrorString(error)
    );


    return output;
}