#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <nn/module.hpp>

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
    std::string code =
        "#pragma once\n\n"
        "#include <nn/modules/linear.hpp>\n"
        "#include <nn/modules/relu.hpp>\n"
        "#include <nn/modules/sigmoid.hpp>\n"
        "#include <nn/sequential.hpp>\n\n"
        "template <typename T>"
        "class " +
        model_name +
        " {\n"
        "public:\n"
        " static nn::Sequential<T> create()\n"
        " {\n"
        "   auto model = nn::Sequential<T>();\n";

    for (auto& m : modules_) {
      code += "   model.add(" + m->save(model_name) + ");\n";
    }
    code +=
        "   return model;\n"
        " }\n"
        "};\n";

    std::string dest = "../trained_models/" + model_name + ".hpp";
    std::ofstream dest_file;
    dest_file.open(dest);
    dest_file << code;
    dest_file.close();
    std::cout << "saved to " << dest << std::endl;
    return code;
  };

 private:
  std::vector<std::shared_ptr<Module<T>>> modules_;
};

}  // namespace nn
