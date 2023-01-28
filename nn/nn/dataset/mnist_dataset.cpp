#include <endian.h>

#include <fstream>
#include <iostream>
#include <nn/dataset/mnist_dataset.hpp>
#include <string>

namespace nn {

MnistDataset::MnistDataset(const fs::path& path, Set set) : path_(path), set_(set) {
  std::string type = set == Set::TRAIN ? "train" : "t10k";

  auto images_file_path = path / (type + "-images.idx3-ubyte");
  images_ = load_data(images_file_path);

  auto labels_file_path = path / (type + "-labels.idx1-ubyte");
  labels_ = load_data(labels_file_path);

  assert(images_.size() == labels_.size() && "Images and labels need to be the same size.");
}

std::vector<Matrix<double>> MnistDataset::load_data(const fs::path& file_path) const {
  // https://www.fon.hum.uva.nl/praat/manual/IDX_file_format.html
  auto file = std::ifstream(file_path, std::ios::binary);

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

  std::vector<Matrix<double>> result;
  result.reserve(count);

  for (std::uint32_t i = 0; i < count; i++) {
    std::vector<double> data;
    data.reserve(nelements);

    for (std::uint32_t j = 0; j < nelements; j++) {
      std::uint8_t byte;
      file.read(reinterpret_cast<char*>(&byte), sizeof(byte));
      data.push_back(static_cast<double>(byte));
    }

    result.emplace_back(Matrix<double>(static_cast<std::size_t>(size),
                                       static_cast<std::size_t>(size), std::move(data)));
  }

  return result;
}

// TODO: Think about what to return here, it's not really nice to always copy the image data
std::pair<Matrix<double>, Matrix<double>> MnistDataset::get(std::size_t idx) const {
  return std::make_pair(images_[idx], labels_[idx]);
}

std::size_t MnistDataset::size() const { return images_.size(); };

}  // namespace nn
