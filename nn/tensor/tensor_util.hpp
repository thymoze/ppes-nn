#pragma once

#include <sstream>
#include <vector>

using Strides = std::vector<std::size_t>;
using Shape = std::vector<std::size_t>;
using Indices = std::vector<std::size_t>;

std::string to_string(std::vector<std::size_t> x) {
  std::stringstream s;
  s << "(";
  std::copy(x.begin(), x.end(), std::ostream_iterator<std::size_t>(s, ", "));
  s << ")";
  return s.str();
}

template <typename T>
class Tensor;

template <typename T>
class TensorData;

template <typename T>
class TensorBackend;

template <typename T>
class Function;

template <typename T>
class SimpleOps;

template <typename T>
struct History;

template <typename T>
class Context;
