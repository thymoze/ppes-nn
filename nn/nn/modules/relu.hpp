#pragma once

#include <nn/module.hpp>

namespace nn {

class ReLU : public Module {
 public:
  ReLU() = default;

  std::vector<Variable<double>> forward(const std::vector<Variable<double>>& inputs) override;

  std::string save(const std::string& model_name) override;

 private:
};

}  // namespace nn
