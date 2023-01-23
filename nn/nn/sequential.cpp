#include "sequential.hpp"

#include <iostream>
#include <iostream>
#include <fstream>

namespace nn {

Sequential::Sequential() = default;

std::vector<Variable<double>> Sequential::forward(const std::vector<Variable<double>>& inputs) {
  auto out = inputs;
  for (auto& module : _modules) {
    out = module->forward(out);
  }
  return {out};
}

std::string Sequential::save(std::string model_name) {

  std::string code =
    "#pragma once\n\n"
    "#include <nn/sequential.hpp>\n"
    "#include <nn/modules/linear.hpp>\n"
    "#include <nn/modules/relu.hpp>\n"
    "#include <nn/modules/sigmoid.hpp>\n\n"
    "class " +
    model_name +
    "\n"
    "{\n"
    "public:\n"
    " static nn::Sequential create()\n"
    " {\n"
    "   auto model = nn::Sequential();\n";

  for (auto &m : _modules) {
    code += "   model.add(" + m->save(model_name) + ");\n";
  }
  code += "   return model;\n"
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

}  // namespace nn
