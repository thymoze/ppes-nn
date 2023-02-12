#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <nn/module.hpp>
#include <sstream>

namespace nn {

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

  std::vector<Variable<T>> forward(const std::vector<Variable<T>>& inputs) override {
    auto out = inputs;
    for (auto& module : modules_) {
      out = module->forward(out);
    }
    return {out};
  };

  std::string save(const std::string& model_name) override {
    std::stringstream code_stream;
    code_stream << "#pragma once" << std::endl
                << std::endl
                << "#include <nn/modules/linear.hpp>" << std::endl
                << "#include <nn/modules/relu.hpp>" << std::endl
                << "#include <nn/modules/sigmoid.hpp>" << std::endl
                << "#include <nn/sequential.hpp>" << std::endl
                << std::endl
                << "template <typename T>" << std::endl
                << "class " << model_name << " {" << std::endl
                << "public:" << std::endl
                << "  static nn::Sequential<T> create() {" << std::endl
                << "    auto model = nn::Sequential<T>();" << std::endl;
    for (auto& m : modules_) {
      code_stream << "    model.add(" << m->save(model_name) << ");" << std::endl;
    }
    code_stream << "    return model;" << std::endl << "  }" << std::endl << "};" << std::endl;
    std::stringstream dest;
    dest << "../../trained_models/" << model_name << ".hpp";

    std::ofstream dest_file;
    dest_file.open(dest.str());
    dest_file << code_stream.str();
    dest_file.close();
    std::cout << "saved to " << dest.str() << std::endl;
    return code_stream.str();
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
