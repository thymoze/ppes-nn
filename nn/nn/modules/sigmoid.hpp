#pragma once

#include <nn/module.hpp>
#include <vector>

namespace nn {

class Sigmoid : public Module {
 public:
  Sigmoid();

  std::vector<Variable<double>> forward(const std::vector<Variable<double>>& inputs) override;

  std::string save(std::string model_name) override;

 private:
};

}  // namespace nn