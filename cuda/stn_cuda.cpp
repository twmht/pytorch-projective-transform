#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> stn_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    int output_h,
    int output_w
    );

std::vector<torch::Tensor> stn_cuda_backward(
    torch::Tensor input,
    torch::Tensor source,
    torch::Tensor grad_input);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> stn_forward(
    torch::Tensor input,
    torch::Tensor weights,
    int output_h,
    int output_w
    ) {
  CHECK_INPUT(input);
  CHECK_INPUT(weights);

  return stn_cuda_forward(input, weights, output_h, output_w);
}

std::vector<torch::Tensor> stn_backward(
    torch::Tensor input,
    torch::Tensor source,
    torch::Tensor grad_input
    ) {
  CHECK_INPUT(input);
  CHECK_INPUT(source);
  CHECK_INPUT(grad_input);

  return stn_cuda_backward(
      input,
      source,
      grad_input
      );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &stn_forward, "STN forward (CUDA)");
  m.def("backward", &stn_backward, "STN backward (CUDA)");
}
