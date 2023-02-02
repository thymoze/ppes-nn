#pragma once

#include <random>

namespace nn {
namespace random {

std::random_device rd;
std::mt19937 gen(rd());

void seed(std::mt19937::result_type seed) { gen.seed(seed); }

double rand(double low, double hi) {
  std::uniform_real_distribution<> dis(low, hi);
  return dis(gen);
}

}  // namespace random
}  // namespace nn
