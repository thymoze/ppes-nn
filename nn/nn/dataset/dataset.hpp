#pragma once

#include <iterator>
#include <utility>
#include <vector>

namespace nn {

template <typename S, typename L>
class Dataset {
 public:
  virtual ~Dataset() = default;

  virtual std::pair<S, L> get(std::size_t idx) const = 0;
  virtual std::size_t size() const = 0;

  class Iterator {
   public:
    // clang-format off
    using iterator_category = std::forward_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = std::pair<S, L>;
    using pointer           = value_type*;
    using reference         = value_type&;
    // clang-format on

    Iterator() : dataset_(nullptr), idx_(-1) {}

    explicit Iterator(Dataset* dataset) : dataset_(dataset), idx_(dataset_->size() > 0 ? 0 : -1) {}

    value_type operator*() {
      return dataset_->get(idx_);
      // return buffer_;
    }

    // Prefix increment
    Iterator& operator++() {
      if (++idx_ >= static_cast<int>(dataset_->size())) {
        idx_ = -1;
      }
      return *this;
    }

    // Postfix increment
    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    bool operator==(const Iterator& that) { return this->idx_ == that.idx_; };
    bool operator!=(const Iterator& that) { return this->idx_ != that.idx_; };

   private:
    Dataset* dataset_;
    int idx_;
    // value_type buffer_;
  };

  Iterator begin() { return Iterator(this); }

  Iterator end() { return Iterator(); }

 protected:
  Dataset() = default;
};

}  // namespace nn
