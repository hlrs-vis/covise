#pragma once
#include "object.h"

namespace core::simulation {
class DataStorage {
 public:
  void addData(const std::string &key, const std::vector<double> &value) {
    m_data[key] = value;
  }

  void addData(const std::string &key, const double &value) {
    m_data[key].push_back(value);
  }
  auto &getData() { return m_data; }

 private:
  // general meta data for the simulation
  Data m_data;
};
}  // namespace core::simulation
