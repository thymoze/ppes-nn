#pragma once

#include <vector>

#include "autograd/variable.hpp"

namespace nn {

template <typename T>
class Module {
 public:
  virtual ~Module() = default;

  virtual std::vector<Variable<T>> forward(const std::vector<Variable<T>>& inputs) = 0;

  std::vector<Variable<T>>& params() { return params_; }

  std::vector<Variable<T>> operator()(const std::vector<Variable<T>>& inputs) {
    return forward(inputs);
  }

  void zero_grad() {
    for (auto& param : params_) {
      param.zero_grad();
    }
  }

  virtual std::string save(const std::string& model_name) = 0;

 protected:
  std::vector<Variable<T>> params_;

  Module() = default;
};

}  // namespace nn
