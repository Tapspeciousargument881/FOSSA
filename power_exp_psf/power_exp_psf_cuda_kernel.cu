#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {
template <typename scalar_t>
__device__ __forceinline__ scalar_t power_exp_psf(scalar_t c, scalar_t x, scalar_t y, float p) {
    float r2 = x * x + y * y;           // >= 0
    float inv_c2 = 1.0f / (c * c);      // assumes c > 0
    float term = powf(r2 * inv_c2, 0.5f * p);  // (r2 / c^2)^(p/2)
    return inv_c2 * expf(-2.0f * term);
}

template <typename scalar_t>
__global__ void power_exp_psf_cuda_forward_kernel(
    const scalar_t* __restrict__ vInput,
    const float p,
    const scalar_t* __restrict__ vWeights,
    const scalar_t* __restrict__ G_x,
    const scalar_t* __restrict__ G_y,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ wsum,
    size_t batch,
    size_t height,
    size_t width,
    size_t kernel_size) {

  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int index = blockIdx.y * width * height + column;

  auto kernel_sum = 0.0;
  auto value = 0.0;
  const int mid = kernel_size / 2;
  const int n = index / height / width;
  const int h = (index / width) % height;
  const int w = index % width;

  if (n < batch && h < height && w < width) {
    if (vWeights[index] > 1) {
        for (int i=0; i<kernel_size; i++) {
            if ((h + i - mid) >= 0 && (h + i - mid) < height) {
                for (int j=0; j<kernel_size; j++) {
                    if ((w + j - mid) >= 0 && (w + j - mid) < width) {
                        if (vWeights[n*width*height + (h+ i - mid)*width + w + j - mid] > 1){
                            const auto g = power_exp_psf(vWeights[n*width*height + (h+ i - mid)*width + w + j - mid],
                                                     G_x[i * kernel_size + j],
                                                     G_y[i * kernel_size + j],
                                                     p);
                            kernel_sum += g;
                            value += g * vInput[n*width*height + (h+ i - mid)*width + w + j - mid];
                        }
                    }
                }
            }
        }
        output[index] = value / kernel_sum;
        wsum[index] = kernel_sum;
    } else {
        output[index] = vInput[index];
    }
  }
}


} // namespace

std::vector<at::Tensor> power_exp_psf_cuda_forward(
    at::Tensor input,
    double p,
    at::Tensor weights,
    at::Tensor G_x,
    at::Tensor G_y) {

  const auto kernel_size = G_x.size(0);
  const auto batch = input.size(0);
  const auto channel = input.size(1);
  const auto height = input.size(2);
  const auto width = input.size(3);
  auto vInput = input.view({-1, height, width});
  auto vWeights = weights.view({-1, height, width});

  const auto batch_size = vInput.size(0);

  auto output = at::zeros_like(vInput);
  auto wsum = at::ones_like(vWeights);

  const int threads = 1024;
  const dim3 blocks((height * width + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(vInput.scalar_type(), "power_exp_psf_forward_cuda", ([&] {
    power_exp_psf_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        vInput.data_ptr<scalar_t>(),
        p,
        vWeights.data_ptr<scalar_t>(),
        G_x.data_ptr<scalar_t>(),
        G_y.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        wsum.data_ptr<scalar_t>(),
        batch_size,
        height,
        width,
        kernel_size);
  }));

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  
  output = output.view({batch, channel, height, width});
  wsum = wsum.view({batch, channel, height, width});

  return {output, wsum};
}