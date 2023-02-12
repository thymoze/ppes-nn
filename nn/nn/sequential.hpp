#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <nn/module.hpp>
#include <sstream>

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

  std::vector<std::shared_ptr<Module<T>>>& modules() { return modules_; }

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

  int prune_one_neuron() override {
    std::vector<std::shared_ptr<Module<T>>> linear_bla;
    for (auto& m : modules_) {
      if (m->is_linear()) {
        linear_bla.push_back(m);
      }
    }

    std::vector<std::pair<std::shared_ptr<Module<T>>, std::shared_ptr<Module<T>>>> linear_layer;

    for (std::size_t i = 1; i < linear_bla.size(); ++i) {
      if (linear_bla[i]->is_prunable()) {
        linear_layer.push_back(std::pair{linear_bla[i], linear_bla[i - 1]});
      }
    }

    std::vector<T> lowest_row_sums;
    for (std::size_t i = 0; i < linear_layer.size(); ++i) {
      std::pair<std::shared_ptr<Module<T>>, std::shared_ptr<Module<T>>> bla = linear_layer[i];
      lowest_row_sums.push_back(bla.first->params()[0].value().lowest_row_sum());
    }

    auto result = std::min_element(lowest_row_sums.begin(), lowest_row_sums.end());
    auto lowest_neuron_layer_index = std::distance(lowest_row_sums.begin(), result);

    auto pruned_neuron = linear_layer[lowest_neuron_layer_index].first->prune_one_neuron();

    linear_layer[lowest_neuron_layer_index].second->apply_pruned_neuron(pruned_neuron);

    return pruned_neuron;
  }

  void apply_pruned_neuron(int) override{};

  bool is_prunable() override { return false; }
  bool is_linear() override { return false; }

 private:
  std::vector<std::shared_ptr<Module<T>>> modules_;
};

}  // namespace nn
