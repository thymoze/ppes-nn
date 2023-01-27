#pragma once

#include <vector>

#include "autograd/variable.hpp"

class Module {
 public:
  virtual ~Module() = default;

  virtual std::vector<Variable<double>> forward(const std::vector<Variable<double>>& inputs) = 0;

  std::vector<Variable<double>>& params() { return _params; }

  std::vector<Variable<double>> operator()(const std::vector<Variable<double>>& inputs) {
    return forward(inputs);
  }

  void zero_grad() {
    for (auto& param : _params) {
      param.zero_grad();
    }
  }

  virtual std::string save(const std::string& model_name) = 0;

 protected:
  std::vector<Variable<double>> _params;

  Module() = default;
};
