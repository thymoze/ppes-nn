#pragma once

#include <memory>
#include <tensor/tensor_util.hpp>

template <typename T>
class Function {
 protected:
  Function() = default;
  [[nodiscard]] virtual std::unique_ptr<Function<T>> current_fn() const = 0;

 public:
  virtual ~Function() = default;

  [[nodiscard]] virtual Tensor<T> forward(Context<T>& ctx,
                                          const std::vector<Tensor<T>>& inputs) const = 0;
  [[nodiscard]] virtual std::vector<Tensor<T>> backward(const Context<T>& ctx,
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

    auto ctx = Context<T>(!need_grad);

    auto res = forward(ctx, raw_vals);

    if (need_grad) {
      auto hist =
          History<T>{current_fn(), std::make_unique<Context<T>>(std::move(ctx)), std::move(inputs)};
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

  [[nodiscard]] Tensor<T> forward(Context<T>& ctx,
                                  const std::vector<Tensor<T>>& inputs) const override {
    assert(inputs.size() == 1 && "Copy expects 1 input.");
    return inputs[0].f().id_map(inputs[0]);
  }

  [[nodiscard]] std::vector<Tensor<T>> backward(const Context<T>& ctx,
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

  [[nodiscard]] Tensor<T> forward(Context<T>& ctx,
                                  const std::vector<Tensor<T>>& inputs) const override {
    assert(inputs.size() == 2 && "View expects 2 inputs.");
    assert(inputs[0].data().is_contiguous() && "Tensor must be contiguous to view.");
    std::vector<T> orig_shape(inputs[0].shape().size());
    std::transform(inputs[0].shape().begin(), inputs[0].shape().end(), orig_shape.begin(),
                   [](auto v) { return static_cast<T>(v); });
    ctx.save_for_backward({Tensor<T>::make(std::move(orig_shape))});

    Shape shape(inputs[1].size());
    std::transform(inputs[1].indices().begin(), inputs[1].indices().end(), shape.begin(),
                   [&inputs](auto i) { return static_cast<std::size_t>(inputs[1][i]); });

    auto data = TensorData<T>(inputs[0].data().data(), std::move(shape));
    return Tensor<T>(std::move(data), inputs[0].f());
  }

  [[nodiscard]] std::vector<Tensor<T>> backward(const Context<T>& ctx,
                                                const Tensor<T>& grad) const override {
    auto& saved = ctx.saved_values();
    Shape shape(saved[0].size());
    std::transform(saved[0].indices().begin(), saved[0].indices().end(), shape.begin(),
                   [&saved](auto i) { return static_cast<std::size_t>(saved[0][i]); });
    auto data = TensorData<T>(grad.data().data(), std::move(shape));
    return {Tensor<T>(std::move(data), grad.f()), Tensor<T>::make(0)};
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

  [[nodiscard]] Tensor<T> forward(Context<T>& ctx,
                                  const std::vector<Tensor<T>>& inputs) const override {
    assert(inputs.size() == 1 && "Negation expects 1 input.");
    return inputs[0].f().neg_map(inputs[0]);
  }

  [[nodiscard]] std::vector<Tensor<T>> backward(const Context<T>& ctx,
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

  [[nodiscard]] Tensor<T> forward(Context<T>& ctx,
                                  const std::vector<Tensor<T>>& inputs) const override {
    assert(inputs.size() == 1 && "Inverse expects 1 input.");
    ctx.save_for_backward(inputs);
    return inputs[0].f().inv_map(inputs[0]);
  }

  [[nodiscard]] std::vector<Tensor<T>> backward(const Context<T>& ctx,
                                                const Tensor<T>& grad) const override {
    auto& saved = ctx.saved_values();
    return {grad.f().inv_back_zip(saved[0], grad)};
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

  [[nodiscard]] Tensor<T> forward(Context<T>& ctx,
                                  const std::vector<Tensor<T>>& inputs) const override {
    assert(inputs.size() == 1 && "ReLU expects 1 input.");
    ctx.save_for_backward(inputs);
    return inputs[0].f().relu_map(inputs[0]);
  }

  [[nodiscard]] std::vector<Tensor<T>> backward(const Context<T>& ctx,
                                                const Tensor<T>& grad) const override {
    auto& saved = ctx.saved_values();
    auto ones = Tensor<T>::ones(saved[0].shape(), saved[0].f());
    return {grad.f().mul_zip(grad, grad.f().relu_map(ones))};
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

  [[nodiscard]] Tensor<T> forward(Context<T>& ctx,
                                  const std::vector<Tensor<T>>& inputs) const override {
    assert(inputs.size() == 1 && "Sigmoid expects 1 input.");
    ctx.save_for_backward(inputs);
    return inputs[0].f().sigmoid_map(inputs[0]);
  }

  [[nodiscard]] std::vector<Tensor<T>> backward(const Context<T>& ctx,
                                                const Tensor<T>& grad) const override {
    auto& saved = ctx.saved_values();
    auto& f = grad.f();
    auto sigmoid = f.sigmoid_map(saved[0]);
    auto sigmoid_deriv = f.mul_zip(sigmoid, f.add_zip(1, f.neg_map(sigmoid)));
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

  Tensor<T> forward(Context<T>& ctx, const std::vector<Tensor<T>>& inputs) const override {
    assert(inputs.size() == 2 && "Multiplication expects 2 inputs.");
    ctx.save_for_backward(inputs);
    return inputs[0].f().mul_zip(inputs[0], inputs[1]);
  }

  [[nodiscard]] std::vector<Tensor<T>> backward(const Context<T>& ctx,
                                                const Tensor<T>& grad) const override {
    auto& saved = ctx.saved_values();
    return {grad.f().mul_zip(grad, saved[1]), grad.f().mul_zip(grad, saved[0])};
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

  [[nodiscard]] Tensor<T> forward(Context<T>& ctx,
                                  const std::vector<Tensor<T>>& inputs) const override {
    assert(inputs.size() == 2 && "Addition expects 2 inputs.");
    return inputs[0].f().add_zip(inputs[0], inputs[1]);
  }

  [[nodiscard]] std::vector<Tensor<T>> backward(const Context<T>& ctx,
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

  [[nodiscard]] Tensor<T> forward(Context<T>& ctx,
                                  const std::vector<Tensor<T>>& inputs) const override {
    assert(inputs.size() == 2 && "Sum expects 2 inputs.");
    auto& x = inputs[0];
    auto& dim = inputs[1];
    return x.f().add_reduce(x, static_cast<std::size_t>(dim.item()));
  }

  [[nodiscard]] std::vector<Tensor<T>> backward(const Context<T>& ctx,
                                                const Tensor<T>& grad) const override {
    return {grad, Tensor<T>::make(0)};
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

  [[nodiscard]] Tensor<T> forward(Context<T>& ctx,
                                  const std::vector<Tensor<T>>& inputs) const override {
    assert(inputs.size() == 2 && "MatMul expects 2 inputs.");
    ctx.save_for_backward(inputs);
    return inputs[0].f().matrix_multiply(inputs[0], inputs[1]);
  }

  [[nodiscard]] std::vector<Tensor<T>> backward(const Context<T>& ctx,
                                                const Tensor<T>& grad) const override {
    auto& saved = ctx.saved_values();
    auto& lhs = saved[0];
    auto& rhs = saved[1];

    auto transpose = [](const Tensor<T>& t) {
      auto ndims = t.ndims();
      std::vector<std::size_t> order(ndims);
      std::iota(order.begin(), order.end(), 0);
      auto last = order[ndims - 1];
      order[ndims - 1] = order[ndims - 2];
      order[ndims - 2] = last;

      return Tensor<T>(t.data().permute(order), t.f());
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
