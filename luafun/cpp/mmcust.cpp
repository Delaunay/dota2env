#include <torch/extension.h>

#include <iostream>

#include <vector>

using BinaryOp = std::function<float(float, float)>;
using OpMap = std::unordered_map<std::string, BinaryOp>;

BinaryOp& supported_ops(std::string name) {
    static OpMap binary_fun = {
        {"add", [](float a, float b) {return a + b; }},
        {"mult", [](float a, float b) {return a * b; }},
        {"div", [](float a, float b) {return a / b; }},
        {"mod", [](float a, float b) {return a % b; }},
        {"pow", [](float a, float b) {return std::pow(a, b); }},

        {"gt", [](float a, float b) {return a > b; }},
        {"gte", [](float a, float b) {return a >= b; }},
        {"lt", [](float a, float b) {return a < b; }},
        {"lte", [](float a, float b) {return a <= b; }},

        {"and", [](float a, float b) {return a & b; }},
        {"nand", [](float a, float b) {return !(a & b); }},
        {"or", [](float a, float b) {return a | b; }},
        {"xor", [](float a, float b) {return a ^ b; }},

    };

    return binary_fun[name];
}

BinaryOp& derivatives(std::string name) {
    static OpMap binary_fun = {


    };

    return binary_fun[name];
}

std::vector<at::Tensor> mmcust_forward(
    torch::Tensor input,
    torch::Tensor weights,
    string combine,
    string reduce)
{
    auto cb = supported_ops(combine);
    auto rd = supported_ops(reduce);

    Matrix w(u.rows(), v.cols(), 0);
    for(size_t r = 0; r < w.rows(); r++) {
        for(size_t c = 0; c < w.cols(); c++) {
            for(size_t i = 0; i < u.cols(); ++i) {

                w(r, c) = rd(w(r, c) , cb(input(r, i), weights(i, c)));

            }
        }
    }

    return {w};
}

std::vector<torch::Tensor> mmcust_backward() {


    return {};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mmcust_forward, "MMCust forward");
  m.def("backward", &mmcust_backward, "MMCust backward");
}
