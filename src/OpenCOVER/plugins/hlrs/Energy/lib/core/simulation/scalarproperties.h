#pragma once
#include <iostream>
#include <map>
#include <string>
#include <utility>

#include "unitmap.h"

namespace core::simulation {

struct ScalarProperty {
  double min;
  double max;
  size_t timesteps;
  std::string unit;
  std::string species;
  std::string preferredColorMap;
};

class ScalarProperties {
 public:
  auto begin() const { return m_properties.begin(); }
  auto end() const { return m_properties.end(); }

  auto getMax(const std::string &key) const { return get(key).max; }
  auto getMin(const std::string &key) const { return get(key).min; }
  auto getMinMax(const std::string &key) const {
    return std::make_pair(getMin(key), getMax(key));
  }
  auto getTimesteps(const std::string &key) const { return get(key).timesteps; }
  auto getSpecies(const std::string &key) const { return get(key).species; }
  auto getUnit(const std::string &key) const { return get(key).unit; }
  auto getPreferredColorMap(const std::string &key) const {
    return get(key).preferredColorMap;
  }

  void setSpecies(const std::string &key, const std::string &species) {
    m_properties[key].species = species;
  }
  void setUnit(const std::string &key) { m_properties[key].unit = UNIT_MAP[key]; }
  void setPreferredColorMap(const std::string &key) {
    m_properties[key].preferredColorMap = COLORMAP_MAP[key];
  }

  auto &ref() { return m_properties; }

 private:
  const ScalarProperty &get(const std::string &key) const {
    auto it = m_properties.find(key);
    if (it == m_properties.end())
      throw std::out_of_range("Key not found Simulation: " + key);
    return it->second;
  }
  std::map<std::string, ScalarProperty> m_properties;
};
}  // namespace core::simulation
