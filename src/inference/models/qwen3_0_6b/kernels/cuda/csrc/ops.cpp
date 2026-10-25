#include <torch/extension.h>
#include <torch/library.h>

torch::Tensor paged_decode_cuda(
    const torch::Tensor& query,
    const torch::Tensor& kv_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lengths,
    int64_t layer_index,
    int64_t block_size_tokens,
    double scale);

TORCH_LIBRARY(qwen3_cuda, library) {
  library.def(
      "paged_decode(Tensor query, Tensor kv_cache, Tensor block_tables, "
      "Tensor context_lengths, int layer_index, int block_size_tokens, "
      "float scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(qwen3_cuda, CUDA, library) {
  library.impl("paged_decode", paged_decode_cuda);
}
