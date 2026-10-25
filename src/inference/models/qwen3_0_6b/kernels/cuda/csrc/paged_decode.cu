#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <math_constants.h>

namespace {

template <typename scalar_t>
__global__ void paged_decode_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ kv_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ context_lengths,
    scalar_t* __restrict__ output,
    int batch_size,
    int query_heads,
    int kv_heads,
    int head_dim,
    int block_size,
    int table_width,
    int num_blocks,
    int layer_index,
    float scale) {
  const int batch = blockIdx.x;
  const int query_head = blockIdx.y;
  const int dim = threadIdx.x;
  const int group_size = query_heads / kv_heads;
  const int kv_head = query_head / group_size;
  const int context_length = context_lengths[batch];
  const scalar_t* q = query + (static_cast<int64_t>(batch) * query_heads + query_head) * head_dim;
  scalar_t* out = output + (static_cast<int64_t>(batch) * query_heads + query_head) * head_dim;

  __shared__ float reduction[128];
  __shared__ float running_max;
  __shared__ float running_sum;
  __shared__ float old_scale;
  __shared__ float new_weight;

  if (dim == 0) {
    running_max = -CUDART_INF_F;
    running_sum = 0.0f;
  }
  __syncthreads();

  float accumulator = 0.0f;
  const int64_t layer_stride =
      static_cast<int64_t>(2) * num_blocks * block_size * kv_heads * head_dim;
  const int64_t key_base = static_cast<int64_t>(layer_index) * layer_stride;
  const int64_t value_base = key_base +
      static_cast<int64_t>(num_blocks) * block_size * kv_heads * head_dim;

  for (int position = 0; position < context_length; ++position) {
    const int logical_block = position / block_size;
    const int physical_block = block_tables[static_cast<int64_t>(batch) * table_width + logical_block];
    const int block_offset = position % block_size;
    const int64_t kv_offset =
        (static_cast<int64_t>(physical_block) * block_size + block_offset) * kv_heads * head_dim +
        static_cast<int64_t>(kv_head) * head_dim;
    const scalar_t* key = kv_cache + key_base + kv_offset;
    const scalar_t* value = kv_cache + value_base + kv_offset;

    reduction[dim] = dim < head_dim
        ? static_cast<float>(q[dim]) * static_cast<float>(key[dim])
        : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (dim < stride) {
        reduction[dim] += reduction[dim + stride];
      }
      __syncthreads();
    }

    if (dim == 0) {
      const float score = reduction[0] * scale;
      const float next_max = fmaxf(running_max, score);
      old_scale = isinf(running_max) ? 0.0f : expf(running_max - next_max);
      new_weight = expf(score - next_max);
      running_sum = running_sum * old_scale + new_weight;
      running_max = next_max;
    }
    __syncthreads();
    if (dim < head_dim) {
      accumulator = accumulator * old_scale + new_weight * static_cast<float>(value[dim]);
    }
    __syncthreads();
  }

  if (dim < head_dim) {
    out[dim] = static_cast<scalar_t>(accumulator / running_sum);
  }
}

int next_power_of_two(int value) {
  int result = 1;
  while (result < value) {
    result <<= 1;
  }
  return result;
}

}  // namespace

torch::Tensor paged_decode_cuda(
    const torch::Tensor& query,
    const torch::Tensor& kv_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lengths,
    int64_t layer_index,
    int64_t block_size_tokens,
    double scale) {
  TORCH_CHECK(query.is_cuda() && kv_cache.is_cuda(), "query and KV cache must be CUDA tensors");
  TORCH_CHECK(query.device() == kv_cache.device(), "query and KV cache must use the same CUDA device");
  TORCH_CHECK(query.scalar_type() == kv_cache.scalar_type(), "query and KV cache dtypes must match");
  TORCH_CHECK(query.is_contiguous() && kv_cache.is_contiguous(), "query and KV cache must be contiguous");
  TORCH_CHECK(block_tables.is_cuda() && context_lengths.is_cuda(), "attention metadata must be CUDA tensors");
  TORCH_CHECK(block_tables.device() == query.device() && context_lengths.device() == query.device(), "metadata device mismatch");
  TORCH_CHECK(block_tables.dim() == 2 && context_lengths.dim() == 1, "unexpected attention metadata rank");
  TORCH_CHECK(block_tables.scalar_type() == at::kInt && context_lengths.scalar_type() == at::kInt, "metadata must use int32");
  TORCH_CHECK(block_tables.is_contiguous() && context_lengths.is_contiguous(), "metadata must be contiguous");
  TORCH_CHECK(query.dim() == 3 && kv_cache.dim() == 6, "unexpected query or KV cache rank");
  TORCH_CHECK(kv_cache.size(1) == 2, "KV cache axis 1 must contain key and value pages");
  TORCH_CHECK(query.size(0) > 0 && query.size(1) > 0 && query.size(2) > 0, "query dimensions must be positive");
  TORCH_CHECK(kv_cache.size(0) > 0 && kv_cache.size(2) > 0 && kv_cache.size(3) > 0 &&
              kv_cache.size(4) > 0 && kv_cache.size(5) > 0, "KV cache dimensions must be positive");
  TORCH_CHECK(query.size(0) == block_tables.size(0) && query.size(0) == context_lengths.size(0), "batch metadata length mismatch");
  TORCH_CHECK(kv_cache.size(3) == block_size_tokens, "KV page size mismatch");
  TORCH_CHECK(query.size(2) == kv_cache.size(5), "query and KV head dimensions differ");
  TORCH_CHECK(query.size(1) % kv_cache.size(4) == 0, "query heads must be divisible by KV heads");
  TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16 ||
              query.scalar_type() == at::kFloat, "paged decode supports float16, bfloat16, and float32");
  TORCH_CHECK(query.size(2) <= 128 && next_power_of_two(query.size(2)) == query.size(2), "head dimension must be a power of two no larger than 128");
  TORCH_CHECK(layer_index >= 0 && layer_index < kv_cache.size(0), "layer index is outside the KV cache");
  TORCH_CHECK(block_size_tokens > 0 && block_tables.size(1) > 0, "block table must have a positive page width");
  TORCH_CHECK(std::isfinite(scale) && scale > 0.0, "attention scale must be finite and positive");

  c10::cuda::CUDAGuard device_guard(query.device());
  auto output = at::empty_like(query);
  const int batch_size = query.size(0);
  const int query_heads = query.size(1);
  const int head_dim = query.size(2);
  const int kv_heads = kv_cache.size(4);
  const int block_size = block_size_tokens;
  const int table_width = block_tables.size(1);
  const int num_blocks = kv_cache.size(2);
  const dim3 grid(batch_size, query_heads);
  const int threads = next_power_of_two(head_dim);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(query.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, query.scalar_type(), "qwen3_cuda_paged_decode", [&] {
        paged_decode_kernel<scalar_t><<<grid, threads, 0, stream>>>(
            query.data_ptr<scalar_t>(),
            kv_cache.data_ptr<scalar_t>(),
            block_tables.data_ptr<int32_t>(),
            context_lengths.data_ptr<int32_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            query_heads,
            kv_heads,
            head_dim,
            block_size,
            table_width,
            num_blocks,
            layer_index,
            static_cast<float>(scale));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
