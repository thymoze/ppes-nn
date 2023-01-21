#pragma once

#include <nn/module.hpp>

namespace nn {

class Linear : public Module {
public:
    Linear(std::size_t input_size, std::size_t output_size, bool bias = true);

    std::vector<Variable<double>> forward(const std::vector<Variable<double>>& inputs) override;

private:
    std::size_t num_in;
    std::size_t num_out;
    bool bias;
};

}