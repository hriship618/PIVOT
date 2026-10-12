#include <torch/extension.h>


// Implemented in spatial_relations.cu
torch::Tensor spatial_relations_cuda(
    torch::Tensor players
);


torch::Tensor spatial_relations(
    torch::Tensor players
) {
    TORCH_CHECK(
        players.is_cuda(),
        "spatial_relations currently supports CUDA tensors only"
    );

    return spatial_relations_cuda(
        players.contiguous()
    );
}


PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    module
) {
    module.def(
        "spatial_relations",
        &spatial_relations,
        "PIVOT spatial relations CUDA kernel"
    );
}