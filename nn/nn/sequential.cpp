#include "sequential.hpp"

namespace nn {

Sequential::Sequential() = default;

std::vector<Variable<double>> Sequential::forward(const std::vector<Variable<double>>& inputs) {
  auto out = inputs;
  for (auto& module : _modules) {
    out = module->forward(out);
  }
  return {out};
}

}  // namespace nn
