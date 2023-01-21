#include "relu.hpp"

#include <matrix/matrix.hpp>
#include <autograd/autograd.hpp>

namespace nn {

ReLU::ReLU() {}

std::vector<Variable<double>> ReLU::forward(const std::vector<Variable<double>>& inputs) {
    auto x = ag::max(inputs[0], 0.);

    return { x };
}

}