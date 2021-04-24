#include <torch/extension.h>

#include <iostream>

#include <vector>

std::vector<at::Tensor> mmcust_forward() {
}



std::vector<torch::Tensor> mmcust_backward() {
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mmcust_forward, "MMCust forward");
  m.def("backward", &mmcust_backward, "MMCust backward");
}
