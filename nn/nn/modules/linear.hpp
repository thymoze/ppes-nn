#pragma once

#include <nn/module.hpp>

namespace nn {

class Linear : public Module {
 public:
  Linear(std::size_t input_size, std::size_t output_size, bool bias = true);
  Linear(Variable<double> weights, Variable<double> bias_weights);
  Linear(Variable<double> weights);

  std::vector<Variable<double>> forward(const std::vector<Variable<double>>& inputs) override;

  std::string save(std::string model_name) override;

 private:
  std::size_t num_in;
  std::size_t num_out;
  bool bias;
};

}  // namespace nn
