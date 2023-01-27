#pragma once

#include <memory>
#include <nn/module.hpp>

namespace nn {

class Sequential : public Module {
 public:
  Sequential() = default;

  template <typename M>
  void add(const M& module) {
    add(std::make_shared<M>(module));
  }

  template <typename M>
  void add(std::shared_ptr<M> module) {
    modules_.emplace_back(module);
    params_.insert(params_.end(), module->params().begin(), module->params().end());
  }

  std::vector<Variable<double>> forward(const std::vector<Variable<double>>& inputs) override;

  std::string save(const std::string& model_name) override;

 private:
  std::vector<std::shared_ptr<Module>> modules_;
};

}  // namespace nn
