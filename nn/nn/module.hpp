#pragma once

#include <tensor/tensor.hpp>
#include <vector>

namespace nn {

template <typename T>
class Parameter {
 public:
  explicit Parameter(Tensor<T> val) : value_(std::make_shared<Tensor<T>>(std::move(val))) {
    value_->requires_grad(true);
  }

  Tensor<T>& value() const { return *value_; }
  void update(Tensor<T> val) {
    *value_ = std::move(val);
    value_->requires_grad(true);
  }

 private:
  std::shared_ptr<Tensor<T>> value_;
};

template <typename T>
class Module {
 public:
  virtual ~Module() = default;

  virtual Tensor<T> forward(const Tensor<T>& input) = 0;

  std::vector<Parameter<T>>& params() { return params_; }

  Tensor<T> operator()(const Tensor<T>& input) { return forward(input); }

  void zero_grad() {
    for (auto& param : params_) {
      param.value().zero_grad();
    }
  }

  virtual std::string save(const std::string& model_name) = 0;

 protected:
  std::vector<Parameter<T>> params_;

  Module() = default;
};

}  // namespace nn
