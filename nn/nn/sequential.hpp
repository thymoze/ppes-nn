#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <nn/module.hpp>

namespace nn {

using namespace tensor;

template <typename T>
class Sequential : public Module<T> {
 public:
  Sequential() = default;

  template <typename M>
  void add(const M& module) {
    add(std::make_shared<M>(module));
  }

  template <typename M>
  void add(std::shared_ptr<M> module) {
    modules_.emplace_back(module);
    this->params_.insert(this->params_.end(), module->params().begin(), module->params().end());
  }

  void init() override {
    for (auto& module : modules_) {
      module->init();
    }
  }

  unsigned int init(const unsigned char data[], const unsigned int data_len) override {
    unsigned int offset = 0;
    for (auto& module : modules_) {
      offset += module->init(data + offset, data_len - offset);
    }
    return offset;
  }

  Tensor<T> forward(const Tensor<T>& inputs) override {
    auto out = inputs;
    for (auto& module : modules_) {
      out = module->forward(out);
    }
    return out;
  };

 private:
  std::vector<std::shared_ptr<Module<T>>> modules_;
};

}  // namespace nn
