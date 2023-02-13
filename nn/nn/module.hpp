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

  virtual std::vector<std::uint8_t> data() = 0;

  virtual std::string to_string() = 0;

  void save(const std::string& path, const std::string& name) {
    auto stream = std::ofstream(path, std::ios::trunc);
    stream << "#pragma once\n\n";
    stream << "namespace " << name << " {\n\n";
    stream << "alignas(" << std::alignment_of_v<T> << ") ";
    stream << "const unsigned char " << name << "[] = { ";
    auto data = this->data();
    for (auto& v : data) {
      stream << "0x" << std::hex << static_cast<int>(v) << ", ";
    }
    stream.seekp((int)stream.tellp() - 2);
    stream << " };\n";
    stream << "const unsigned int " << name << "_len = " << std::dec << data.size() << ";\n";
    stream << "\n}\n";
  }

  virtual bool is_prunable() = 0;

  bool has_params() const { return params_.size() != 0; }
  virtual int prune_one_neuron() = 0;
  virtual void apply_pruned_neuron(int neuron) = 0;

  virtual bool is_linear() = 0;

 protected:
  std::vector<Parameter> params_;

  Module() = default;
};

}  // namespace nn
