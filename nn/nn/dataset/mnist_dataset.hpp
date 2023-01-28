#pragma once

#include <filesystem>
#include <nn/dataset/dataset.hpp>

namespace fs = std::filesystem;

namespace nn {

/// @brief A dataset wrapper for the MNIST Database: http://yann.lecun.com/exdb/mnist/
class MnistDataset : public Dataset<Matrix<double>, Matrix<double>> {
 public:
  enum class Set { TRAIN, TEST };

  /// @param path Path to the directory containing the *-images.idx3-ubyte and *-labels.idx1-ubyte
  /// files
  /// @param set TRAIN or TEST to read the train-* or t10k-* files
  MnistDataset(const fs::path& path, Set set);

  std::pair<Matrix<double>, Matrix<double>> get(std::size_t idx) const override;
  std::size_t size() const override;

 private:
  MnistDataset() = default;

  std::vector<Matrix<double>> load_data(const fs::path& file_path) const;

  fs::path path_;
  Set set_;
  std::vector<Matrix<double>> images_;
  std::vector<Matrix<double>> labels_;
};

}  // namespace nn
