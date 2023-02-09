#pragma once

#include <fstream>
#include <tensor/tensor.hpp>
#include <vector>

namespace nn {

using namespace tensor;

template <typename T>
class Parameter {
 public:
  Parameter() : value_(std::make_shared<Tensor<T>>()) {}
  explicit Parameter(Tensor<T> val) : value_(std::make_shared<Tensor<T>>(std::move(val))) {
    value_->requires_grad(true);
  }

  Tensor<T>& value() const { return *value_; }
  void update(Tensor<T> val) {
    *value_ = std::move(val);
    value_->requires_grad(true);
  }

 private:
  std::shared_ptr<Tensor<T>> value_;
};

template <typename T>
class Module {
 public:
  virtual ~Module() = default;

  virtual Tensor<T> forward(const Tensor<T>& input) = 0;

  std::vector<Parameter<T>>& params() { return params_; }

  Tensor<T> operator()(const Tensor<T>& input) { return forward(input); }

  void zero_grad() {
    for (auto& param : params_) {
      param.value().zero_grad();
    }
  }

  virtual void init() = 0;

  virtual unsigned int init(const unsigned char data[], const unsigned int data_len) = 0;

  void save(const std::string& path, const std::string& name) {
    std::vector<std::string> param_names;

    auto stream = std::ofstream(path, std::ios::trunc);
    stream << "#pragma once\n\n";
    stream << "namespace " << name << " {\n\n";
    stream << "const unsigned char " << name << "[] = { ";
    int size = 0;
    for (auto& param : params_) {
      auto& data = *param.value().data();
      size += data.size() * sizeof(T);
      for (auto& v : data) {
        auto* p = reinterpret_cast<const std::uint8_t*>(&v);
        for (std::size_t s = 0; s < sizeof(v); ++s) {
          stream << "0x" << std::hex << static_cast<int>(*(p + s)) << ", ";
        }
      }
    }
    stream.seekp((int)stream.tellp() - 2);
    stream << " };\n";
    stream << "const unsigned int " << name << "_len = " << std::dec << size << ";\n";
    stream << "\n}\n";
  }

 protected:
  std::vector<Parameter<T>> params_;

  Module() = default;
};

}  // namespace nn
