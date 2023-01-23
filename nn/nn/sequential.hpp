#pragma once

#include <memory>
#include <nn/module.hpp>

namespace nn {

class Sequential : public Module {
 public:
  Sequential();

  template <typename M>
  void add(const M& module) {
    add(std::make_shared<M>(module));
  }

  template <typename M>
  void add(std::shared_ptr<M> module) {
    _modules.emplace_back(module);
    _params.insert(_params.end(), module->params().begin(), module->params().end());
  }

  std::vector<Variable<double>> forward(const std::vector<Variable<double>>& inputs) override;

 private:
  std::vector<std::shared_ptr<Module>> _modules;
};

}  // namespace nn