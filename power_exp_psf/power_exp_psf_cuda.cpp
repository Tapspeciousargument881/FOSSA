#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

std::vector<at::Tensor> power_exp_psf_cuda_forward(
    at::Tensor input,
    double p,
    at::Tensor weights,
    at::Tensor G_x,
    at::Tensor G_y);


// C++ interface

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda())
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous())
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> power_exp_psf_forward(
    at::Tensor input,
    double p,
    at::Tensor weights,
    at::Tensor G_x,
    at::Tensor G_y) {
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(G_x);
  CHECK_INPUT(G_y);

  return power_exp_psf_cuda_forward(input, p, weights, G_x, G_y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &power_exp_psf_forward, "power_exp_psf forward (CUDA)");
}