#pragma once

#include <any>
#include <fstream>
#include <tensor/tensor.hpp>
#include <vector>

namespace nn {

using namespace tensor;

class Parameter {
 public:
  Parameter() : value_(std::make_shared<std::any>()) {}

  template <typename T>
  T& value() {
    if (typeid(T) == value_->type())
      return *std::any_cast<T>(value_.get());
    else
      throw std::logic_error("Can't request value of type " + std::string(typeid(T).name()) +
                             " from parameter of type " + std::string(value_->type().name()));
  }

  template <typename T>
  void update(T val) {
    val.requires_grad(true);
    *value_ = std::move(val);
  }

 private:
  std::shared_ptr<std::any> value_;
};

template <typename T>
Parameter make_param(Tensor<T> val) {
  Parameter param;
  param.update(val);
  return param;
}

template <typename T>
class Module {
 public:
  virtual ~Module() = default;

  virtual Tensor<T> forward(const Tensor<T>& input) = 0;

  std::vector<Parameter>& params() { return params_; }

  Tensor<T> operator()(const Tensor<T>& input) { return forward(input); }

  void zero_grad() {
    for (auto& param : params_) {
      auto v = param.template value<Tensor<T>>();
      v.zero_grad();
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
      auto& data = *param.value<Tensor<T>>().data();
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
  std::vector<Parameter> params_;

  Module() = default;
};

}  // namespace nn
