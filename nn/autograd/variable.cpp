// #include "variable.hpp"
// #include <unordered_set>

// template<typename T>
// Variable<T>::Variable(double value) :
// shared_value(std::make_shared<double>(value)) {};

// Variable::Variable(double value, std::vector<Variable> inputs, GradFunc
// gradFunc)
//     : shared_value(std::make_shared<double>(value))
// {
//     shared_grad->inputs = std::move(inputs);
//     shared_grad->gradFunc = std::move(gradFunc);
// }

// void Variable::backward() const {
//     add_grad(1);

//     auto dag = build();
//     for (auto iter = dag.rbegin(); iter != dag.rend(); iter++) {
//         if (iter->shared_grad->gradFunc) {
//             iter->shared_grad->gradFunc(iter->shared_grad->inputs,
//             iter->shared_grad->grad);
//         }
//     }
// }

// std::vector<Variable> Variable::build() const {
//     std::unordered_set<SharedGrad*> cache;
//     std::vector<Variable> dag;
//     std::function<void(const Variable&)> recurse;

//     // Topological sort
//     recurse = [&](const Variable& var) {
//         auto id = var.shared_grad.get();
//         if (cache.find(id) != cache.end()) {
//             return;
//         }
//         for (const auto& input : var.shared_grad->inputs) {
//             recurse(input);
//         }
//         cache.insert(id);
//         dag.push_back(var);
//     };

//     recurse(*this);
//     return dag;
// }

// void Variable::add_grad(double grad) const {
//     shared_grad->grad += grad;
// }

// void Variable::zero_grad() const {
//     shared_grad->grad = 0;
// }

// double Variable::grad() const {
//     return shared_grad->grad;
// }

// double Variable::value() const {
//     return *shared_value;
// }
