#pragma once

#include <vector>

using Strides = std::vector<std::size_t>;
using Shape = std::vector<std::size_t>;
using Indices = std::vector<std::size_t>;

template <typename T>
class Tensor;

template <typename T>
class TensorData;

template <typename T>
class TensorBackend;

template<typename T>
class Function;

template <typename T>
class SimpleOps;

template <typename T>
struct History;

template <typename T>
class Context;
