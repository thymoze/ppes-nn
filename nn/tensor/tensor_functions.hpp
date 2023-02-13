#pragma once

#include <memory>
#include <tensor/tensor_util.hpp>

namespace tensor {

template <typename T>
class Function {
 protected:
  Function() = default;
  [[nodiscard]] virtual std::unique_ptr<Function<T>> current_fn() const = 0;

 public:
  virtual ~Function() = default;

  [[nodiscard]] virtual Tensor<T> forward([[maybe_unused]] Context& ctx,
                                          const std::vector<Tensor<T>>& inputs) const = 0;
  [[nodiscard]] virtual std::vector<Tensor<T>> backward([[maybe_unused]] const Context& ctx,
                                                        const Tensor<T>& grad) const = 0;

  template <typename... I>
  [[nodiscard]] Tensor<T> operator()(I... in) {
    std::vector<Tensor<T>> inputs = {in...};
    std::vector<Tensor<T>> raw_vals;
    bool need_grad = false;
    for (auto& i : inputs) {
      if (i.requires_grad()) need_grad = true;
      raw_vals.push_back(i.detach());
    }

    auto ctx = Context(!need_grad);

    auto res = forward(ctx, raw_vals);

    if (need_grad) {
      auto hist =
          History<T>{current_fn(), std::make_unique<Context>(std::move(ctx)), std::move(inputs)};
      return Tensor<T>(std::move(res), std::move(hist));
    } else {
      return res;
    }
  }
};

template <typename T>
class Copy : public Function<T> {
 public:
  Copy() = default;

  [[nodiscard]] Tensor<T> forward([[maybe_unused]] Context& ctx,
                                  const std::vector<Tensor<T>>& inputs) const override {
    assert(inputs.size() == 1 && "Copy expects 1 input.");
    return inputs[0].f().id_map(inputs[0]);
  }

  [[nodiscard]] std::vector<Tensor<T>> backward([[maybe_unused]] const Context& ctx,
                                                const Tensor<T>& grad) const override {
    return {grad};
  }

 private:
  [[nodiscard]] std::unique_ptr<Function<T>> current_fn() const override {
    auto fn = *this;
    return std::make_unique<decltype(fn)>(std::move(fn));
  }
};

template <typename T>
class View : public Function<T> {
 public:
  View() = default;

  [[nodiscard]] Tensor<T> forward([[maybe_unused]] Context& ctx,
                                  const std::vector<Tensor<T>>& inputs) const override {
    assert(inputs.size() == 2 && "View expects 2 inputs.");
    assert(inputs[0].data()->is_contiguous() && "Tensor must be contiguous to view.");
    ctx.save_for_backward(std::vector<Shape>{inputs[0].shape()});

    Shape shape(inputs[1].size());
    std::transform(inputs[1].indices().begin(), inputs[1].indices().end(), shape.begin(),
                   [&inputs](auto i) { return static_cast<std::size_t>(inputs[1][i]); });

    auto data = inputs[0].data()->view(shape);
    return Tensor<T>(std::move(data), inputs[0].f());
  }

  [[nodiscard]] std::vector<Tensor<T>> backward([[maybe_unused]] const Context& ctx,
                                                const Tensor<T>& grad) const override {
    auto& saved = ctx.saved_values();
    Shape shape = std::any_cast<Shape>(saved[0]);
    auto data = grad.data()->view(shape);
    return {Tensor<T>(std::move(data), grad.f()), tensor::make<T>(0)};
  }

 private:
  [[nodiscard]] std::unique_ptr<Function<T>> current_fn() const override {
    auto fn = *this;
    return std::make_unique<decltype(fn)>(std::move(fn));
  }
};

template <typename T>
class Squeeze : public Function<T> {
 public:
  Squeeze() = default;

  [[nodiscard]] Tensor<T> forward([[maybe_unused]] Context& ctx,
                                  const std::vector<Tensor<T>>& inputs) const override {
    assert(inputs.size() == 1 && "Squeeze expects 1 input.");
    ctx.save_for_backward(std::vector<Shape>{inputs[0].shape(), inputs[0].strides()});

    Shape shape;
    Indices strides;
    for (auto i = 0; i < inputs[0].shape().size(); i++) {
      if (inputs[0].shape()[i] != 1) {
        shape.push_back(inputs[0].shape()[i]);
        strides.push_back(inputs[0].strides()[i]);
      }
    }

    auto data = inputs[0].data()->view(std::move(strides), std::move(shape));
    return Tensor<T>(std::move(data), inputs[0].f());
  }

  [[nodiscard]] std::vector<Tensor<T>> backward([[maybe_unused]] const Context& ctx,
                                                const Tensor<T>& grad) const override {
    auto orig_shape = std::any_cast<Shape>(ctx.saved_values()[0]);
    auto orig_strides = std::any_cast<Strides>(ctx.saved_values()[1]);
    auto data = grad.data()->view(std::move(orig_shape), std::move(orig_strides));
    return {Tensor<T>(std::move(data), grad.f())};
  }

 private:
  [[nodiscard]] std::unique_ptr<Function<T>> current_fn() const override {
    auto fn = *this;
    return std::make_unique<decltype(fn)>(std::move(fn));
  }
};

template <typename T>
class Unsqueeze : public Function<T> {
 public:
  Unsqueeze() = default;

  [[nodiscard]] Tensor<T> forward([[maybe_unused]] Context& ctx,
                                  const std::vector<Tensor<T>>& inputs) const override {
    assert(inputs.size() == 2 && "Unsqueeze expects 2 inputs.");
    auto dim = static_cast<std::size_t>(inputs[1].item());
    ctx.save_for_backward(dim);

    auto stride = dim + 1 < inputs[0].strides().size() ? inputs[0].strides()[dim + 1] : 1;

    Shape shape = inputs[0].shape();
    Indices strides = inputs[0].strides();
    shape.insert(shape.begin() + dim, 1);
    strides.insert(strides.begin() + dim, stride);

    auto data = inputs[0].data()->view(std::move(shape), std::move(strides));
    return Tensor<T>(std::move(data), inputs[0].f());
  }

  [[nodiscard]] std::vector<Tensor<T>> backward([[maybe_unused]] const Context& ctx,
                                                const Tensor<T>& grad) const override {
    auto dim = std::any_cast<std::size_t>(ctx.saved_values()[0]);
    Shape shape = grad.shape();
    Indices strides = grad.strides();
    shape.erase(shape.begin() + dim);
    strides.erase(strides.begin() + dim);
    auto data = grad.data()->view(std::move(shape), std::move(strides));
    return {Tensor<T>(std::move(data), grad.f()), tensor::make<T>(0)};
  }

 private:
  [[nodiscard]] std::unique_ptr<Function<T>> current_fn() const override {
    auto fn = *this;
    return std::make_unique<decltype(fn)>(std::move(fn));
  }
};

template <typename T>
class Neg : public Function<T> {
 public:
  Neg() = default;

  [[nodiscard]] Tensor<T> forward([[maybe_unused]] Context& ctx,
                                  const std::vector<Tensor<T>>& inputs) const override {
    assert(inputs.size() == 1 && "Negation expects 1 input.");
    return inputs[0].f().neg_map(inputs[0]);
  }

  [[nodiscard]] std::vector<Tensor<T>> backward([[maybe_unused]] const Context& ctx,
                                                const Tensor<T>& grad) const override {
    return {grad.f().neg_map(grad)};
  }

 private:
  [[nodiscard]] std::unique_ptr<Function<T>> current_fn() const override {
    auto fn = *this;
    return std::make_unique<decltype(fn)>(std::move(fn));
  }
};

template <typename T>
class Inv : public Function<T> {
 public:
  Inv() = default;

  [[nodiscard]] Tensor<T> forward([[maybe_unused]] Context& ctx,
                                  const std::vector<Tensor<T>>& inputs) const override {
    assert(inputs.size() == 1 && "Inverse expects 1 input.");
    ctx.save_for_backward(inputs);
    return inputs[0].f().inv_map(inputs[0]);
  }

  [[nodiscard]] std::vector<Tensor<T>> backward([[maybe_unused]] const Context& ctx,
                                                const Tensor<T>& grad) const override {
    auto& saved = ctx.saved_values();
    return {grad.f().inv_back_zip(std::any_cast<Tensor<T>>(saved[0]), grad)};
  }

 private:
  [[nodiscard]] std::unique_ptr<Function<T>> current_fn() const override {
    auto fn = *this;
    return std::make_unique<decltype(fn)>(std::move(fn));
  }
};

template <typename T>
class ReLU : public Function<T> {
 public:
  ReLU() = default;

  [[nodiscard]] Tensor<T> forward([[maybe_unused]] Context& ctx,
                                  const std::vector<Tensor<T>>& inputs) const override {
    assert(inputs.size() == 1 && "ReLU expects 1 input.");
    auto relu = inputs[0].f().relu_map(inputs[0]);
    ctx.save_for_backward(inputs[0]);
    ctx.save_for_backward(relu);
    return relu;
  }

  [[nodiscard]] std::vector<Tensor<T>> backward([[maybe_unused]] const Context& ctx,
                                                const Tensor<T>& grad) const override {
    auto input = std::any_cast<Tensor<T>>(ctx.saved_values()[0]);
    auto relu = std::any_cast<Tensor<T>>(ctx.saved_values()[1]);
    auto ones = input == relu;
    return {grad.f().mul_zip(grad, grad.f().relu_map(ones))};
  }

 private:
  [[nodiscard]] std::unique_ptr<Function<T>> current_fn() const override {
    auto fn = *this;
    return std::make_unique<decltype(fn)>(std::move(fn));
  }
};

template <typename T>
class Exp : public Function<T> {
 public:
  Exp() = default;

  [[nodiscard]] Tensor<T> forward([[maybe_unused]] Context& ctx,
                                  const std::vector<Tensor<T>>& inputs) const override {
    assert(inputs.size() == 1 && "Exp expects 1 input.");
    auto e = inputs[0].f().exp_map(inputs[0]);
    ctx.save_for_backward(e);
    return e;
  }

  [[nodiscard]] std::vector<Tensor<T>> backward([[maybe_unused]] const Context& ctx,
                                                const Tensor<T>& grad) const override {
    auto saved = std::any_cast<Tensor<T>>(ctx.saved_values()[0]);
    return {grad.f().mul_zip(grad, saved)};
  }

 private:
  [[nodiscard]] std::unique_ptr<Function<T>> current_fn() const override {
    auto fn = *this;
    return std::make_unique<decltype(fn)>(std::move(fn));
  }
};

template <typename T>
class Sigmoid : public Function<T> {
 public:
  Sigmoid() = default;

  [[nodiscard]] Tensor<T> forward([[maybe_unused]] Context& ctx,
                                  const std::vector<Tensor<T>>& inputs) const override {
    assert(inputs.size() == 1 && "Sigmoid expects 1 input.");
    ctx.save_for_backward(inputs);
    return inputs[0].f().sigmoid_map(inputs[0]);
  }

  [[nodiscard]] std::vector<Tensor<T>> backward([[maybe_unused]] const Context& ctx,
                                                const Tensor<T>& grad) const override {
    auto saved = std::any_cast<Tensor<T>>(ctx.saved_values()[0]);
    auto& f = grad.f();
    auto sigmoid = f.sigmoid_map(saved);
    auto sigmoid_deriv = f.mul_zip(sigmoid, f.add_zip(tensor::make<T>(1), f.neg_map(sigmoid)));
    return {f.mul_zip(grad, sigmoid_deriv)};
  }

 private:
  [[nodiscard]] std::unique_ptr<Function<T>> current_fn() const override {
    auto fn = *this;
    return std::make_unique<decltype(fn)>(std::move(fn));
  }
};

template <typename T>
class Mul : public Function<T> {
 public:
  Mul() = default;

  Tensor<T> forward([[maybe_unused]] Context& ctx,
                    const std::vector<Tensor<T>>& inputs) const override {
    assert(inputs.size() == 2 && "Multiplication expects 2 inputs.");
    ctx.save_for_backward(inputs);
    return inputs[0].f().mul_zip(inputs[0], inputs[1]);
  }

  [[nodiscard]] std::vector<Tensor<T>> backward([[maybe_unused]] const Context& ctx,
                                                const Tensor<T>& grad) const override {
    auto saved_lhs = std::any_cast<Tensor<T>>(ctx.saved_values()[0]);
    auto saved_rhs = std::any_cast<Tensor<T>>(ctx.saved_values()[1]);
    return {grad.f().mul_zip(grad, saved_rhs), grad.f().mul_zip(grad, saved_lhs)};
  }

 private:
  [[nodiscard]] std::unique_ptr<Function<T>> current_fn() const override {
    auto fn = *this;
    return std::make_unique<decltype(fn)>(std::move(fn));
  }
};

template <typename T>
class Add : public Function<T> {
 public:
  Add() = default;

  [[nodiscard]] Tensor<T> forward([[maybe_unused]] Context& ctx,
                                  const std::vector<Tensor<T>>& inputs) const override {
    assert(inputs.size() == 2 && "Addition expects 2 inputs.");
    return inputs[0].f().add_zip(inputs[0], inputs[1]);
  }

  [[nodiscard]] std::vector<Tensor<T>> backward([[maybe_unused]] const Context& ctx,
                                                const Tensor<T>& grad) const override {
    return {grad, grad};
  }

 private:
  [[nodiscard]] std::unique_ptr<Function<T>> current_fn() const override {
    auto fn = *this;
    return std::make_unique<decltype(fn)>(std::move(fn));
  }
};

template <typename T>
class Sum : public Function<T> {
 public:
  Sum() = default;

  [[nodiscard]] Tensor<T> forward([[maybe_unused]] Context& ctx,
                                  const std::vector<Tensor<T>>& inputs) const override {
    assert(inputs.size() == 2 && "Sum expects 2 inputs.");
    auto& x = inputs[0];
    auto& dim = inputs[1];
    return x.f().add_reduce(x, static_cast<std::size_t>(dim.item()));
  }

  [[nodiscard]] std::vector<Tensor<T>> backward([[maybe_unused]] const Context& ctx,
                                                const Tensor<T>& grad) const override {
    return {grad, tensor::make<T>(0)};
  }

 private:
  [[nodiscard]] std::unique_ptr<Function<T>> current_fn() const override {
    auto fn = *this;
    return std::make_unique<decltype(fn)>(std::move(fn));
  }
};

template <typename T>
class MatMul : public Function<T> {
 public:
  MatMul() = default;

  [[nodiscard]] Tensor<T> forward([[maybe_unused]] Context& ctx,
                                  const std::vector<Tensor<T>>& inputs) const override {
    assert(inputs.size() == 2 && "MatMul expects 2 inputs.");
    ctx.save_for_backward(inputs);
    return inputs[0].f().matrix_multiply(inputs[0], inputs[1]);
  }

  [[nodiscard]] std::vector<Tensor<T>> backward([[maybe_unused]] const Context& ctx,
                                                const Tensor<T>& grad) const override {
    auto lhs = std::any_cast<Tensor<T>>(ctx.saved_values()[0]);
    auto rhs = std::any_cast<Tensor<T>>(ctx.saved_values()[1]);
    ;

    auto transpose = [](const Tensor<T>& t) {
      auto ndims = t.ndims();
      Shape order(ndims);
      std::iota(order.begin(), order.end(), 0);
      auto last = order[ndims - 1];
      order[ndims - 1] = order[ndims - 2];
      order[ndims - 2] = last;

      return Tensor<T>(t.data()->permute(order), t.f());
    };

    return {
        grad.f().matrix_multiply(grad, transpose(rhs)),
        grad.f().matrix_multiply(transpose(lhs), grad),
    };
  }

 private:
  [[nodiscard]] std::unique_ptr<Function<T>> current_fn() const override {
    auto fn = *this;
    return std::make_unique<decltype(fn)>(std::move(fn));
  }
};

}  // namespace tensor
