#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "object.h"

namespace core::simulation {

class Simulation {
 public:
  Simulation() = default;

  void addData(const std::string &key, const std::vector<double> &value) {
    m_data[key] = value;
  }

  void addData(const std::string &key, const double &value) {
    m_data[key].push_back(value);
  }

  auto &getData() { return m_data; }
  const auto &getMinMax(const std::string &key) const { return m_minMax.at(key); }
  const auto &getTimesteps(const std::string &key) const {
    return m_timesteps.at(key);
  }

  virtual void computeParameters() {};

 protected:
  template <typename T>
  void computeParameter(const ObjectContainer<T> &baseMap) {
    static_assert(std::is_base_of_v<Object, T>, "T must be derived from core::simulation::Object");
    for (const auto &[_, base] : baseMap.get()) {
      const auto &data = base.getData();
      for (const auto &[key, values] : data) {
        computeMinMax(key, values);
        computeMaxTimestep(key, values);
      }
    }
  }

  virtual void computeMinMax(const std::string &key,
                             const std::vector<double> &values);
  virtual void computeMaxTimestep(const std::string &key,
                                  const std::vector<double> &values);

  std::map<std::string, std::pair<double, double>> m_minMax;
  std::map<std::string, size_t> m_timesteps;
  // general meta data for the simulation
  Data m_data;
};
}  // namespace core::simulation
