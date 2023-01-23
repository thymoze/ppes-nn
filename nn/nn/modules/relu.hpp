#pragma once

#include <nn/module.hpp>

namespace nn {

class ReLU : public Module {
 public:
  ReLU();

  std::vector<Variable<double>> forward(const std::vector<Variable<double>>& inputs) override;

 private:
};

}  // namespace nn
