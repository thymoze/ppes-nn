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
    std::vector<std::shared_ptr<Module<T>>> all_linear_layers;
    for (auto& m : modules_) {
      if (m->is_linear()) {
        all_linear_layers.push_back(m);
      }
    }

    std::vector<std::pair<std::shared_ptr<Module<T>>, std::shared_ptr<Module<T>>>>
        prunable_linear_layers;

    for (std::size_t i = 1; i < all_linear_layers.size(); ++i) {
      if (all_linear_layers[i]->is_prunable()) {
        prunable_linear_layers.push_back(std::pair{all_linear_layers[i], all_linear_layers[i - 1]});
      }
    }

    std::vector<T> lowest_row_sums;
    for (std::size_t i = 0; i < prunable_linear_layers.size(); ++i) {
      lowest_row_sums.push_back(
          tensor::min(
              tensor::abssum(
                  prunable_linear_layers[i].first->params()[0].template value<Tensor<T>>(), 0))
              .item());
    }

    auto result = std::min_element(lowest_row_sums.begin(), lowest_row_sums.end());
    auto lowest_neuron_layer_index = std::distance(lowest_row_sums.begin(), result);

    auto pruned_neuron =
        prunable_linear_layers[lowest_neuron_layer_index].first->prune_one_neuron();

    prunable_linear_layers[lowest_neuron_layer_index].second->apply_pruned_neuron(pruned_neuron);

    return pruned_neuron;
  }

  void apply_pruned_neuron(int) override{};

  bool is_prunable() override { return false; }
  bool is_linear() override { return false; }

  void prune(int amount) {
    for (int i = 0; i < amount; ++i) {
      prune_one_neuron();
    }
    std::cout << "Pruned " << amount << " Neurons" << std::endl
              << "Current model structure:" << std::endl
              << to_string() << std::endl;
  }

  std::string to_string() override {
    std::stringstream stream;
    stream << "auto model = nn::Sequential<T>();" << std::endl;
    for (auto& m : modules_) {
      stream << "model.add(" << m->to_string() << ");" << std::endl;
    }
    return stream.str();
  }

  std::vector<std::uint8_t> data() override {
    std::vector<std::uint8_t> data;
    for (auto& mod : modules_) {
      auto mod_data = mod->data();
      data.insert(data.end(), mod_data.begin(), mod_data.end());
    }
    return data;
  }

 private:
  std::vector<std::shared_ptr<Module<T>>> modules_;
};

}  // namespace nn
