#pragma once

#include <endian.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nn/dataset/dataset.hpp>
#include <nn/dataset/mnist_dataset.hpp>
#include <string>
#include <tensor/tensor.hpp>

namespace fs = std::filesystem;

namespace nn {

/// @brief A dataset wrapper for the MNIST Database: http://yann.lecun.com/exdb/mnist/
template <typename T>
class MnistDataset : public Dataset<Tensor<T>, Tensor<T>> {
 public:
  enum class Set { TRAIN, TEST };

  /// @param path Path to the directory containing the *-images.idx3-ubyte and *-labels.idx1-ubyte
  /// files
  /// @param set TRAIN or TEST to read the train-* or t10k-* files
  MnistDataset(const fs::path& path, Set set, int batch_size) : batch_size_(batch_size) {
    std::string type = set == Set::TRAIN ? "train" : "t10k";

    auto images_file_path = path / (type + "-images-idx3-ubyte");
    auto images_file = std::ifstream(images_file_path, std::ios::binary);
    images_ = load_data(images_file);

    auto labels_file_path = path / (type + "-labels-idx1-ubyte");
    auto labels_file = std::ifstream(labels_file_path, std::ios::binary);
    labels_ = load_data(labels_file);

    assert(images_.size() == labels_.size() && "Images and labels need to be the same size.");
  };

  struct membuf : std::streambuf {
    membuf(char* begin, char* end) { this->setg(begin, begin, end); }
  };

  MnistDataset(std::uint8_t* img_start, std::uint8_t* img_end, std::uint8_t* label_start,
               std::uint8_t* label_end, int batch_size)
      : batch_size_(batch_size) {
    membuf img_buf(reinterpret_cast<char*>(img_start), reinterpret_cast<char*>(img_end));
    std::istream img_stream(&img_buf);
    images_ = load_data(img_stream);

    membuf label_buf(reinterpret_cast<char*>(label_start), reinterpret_cast<char*>(label_end));
    std::istream label_stream(&label_buf);
    labels_ = load_data(label_stream);

    assert(images_.size() == labels_.size() && "Images and labels need to be the same size.");
  }

  std::pair<Tensor<T>, Tensor<T>> get(std::size_t idx) const override {
    if (batch_size_ == 1) {
      return std::pair(images_[idx], labels_[idx]);
    } else {
      auto imgs = stack<T>(images_.begin() + (idx * batch_size_),
                        std::min(images_.begin() + ((idx + 1) * batch_size_), images_.end()));

      auto lbls = stack<T>(labels_.begin() + (idx * batch_size_),
                        std::min(labels_.begin() + ((idx + 1) * batch_size_), labels_.end()));

      return std::pair(imgs, lbls);
    }
  };

  std::size_t size() const override { return images_.size(); }
  int batch_size() const { return batch_size_; }

 private:
  MnistDataset() = default;

  std::vector<Tensor<T>> load_data(std::istream& file) const {
    // https://www.fon.hum.uva.nl/praat/manual/IDX_file_format.html

    std::uint16_t _magic;
    std::uint8_t data_type;
    std::uint8_t ndim;
    std::uint32_t count;
    file.read(reinterpret_cast<char*>(&_magic), sizeof(_magic));
    file.read(reinterpret_cast<char*>(&data_type), sizeof(data_type));
    file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
    file.read(reinterpret_cast<char*>(&count), sizeof(count));
    count = be32toh(count);

    assert(_magic == 0 && "First two magic bytes must be 0.");
    assert(data_type == 8 && "Expected data type ubyte.");
    assert((ndim == 1 || ndim == 3) && "Exptected number of dimensions to be either 1 or 3.");

    std::uint32_t size = 1;
    if (ndim == 3) {
      std::uint32_t dim2;
      file.read(reinterpret_cast<char*>(&size), sizeof(size));
      file.read(reinterpret_cast<char*>(&dim2), sizeof(dim2));
      size = be32toh(size);
      dim2 = be32toh(dim2);
      assert(size == dim2 && "Expected square dimensions.");
    }

    auto nelements = size * size;

    std::vector<Tensor<T>> result;
    result.reserve(count);

    for (std::uint32_t i = 0; i < count; i++) {
      std::vector<T> data;
      data.reserve(nelements);

      for (std::uint32_t j = 0; j < nelements; j++) {
        std::uint8_t byte;
        file.read(reinterpret_cast<char*>(&byte), sizeof(byte));
        data.push_back(static_cast<T>(byte));
      }

      result.emplace_back(Tensor<T>::make({size, size}, std::move(data)));
    }

    return result;
  };

  int batch_size_;
  std::vector<Tensor<T>> images_;
  std::vector<Tensor<T>> labels_;
};

}  // namespace nn
