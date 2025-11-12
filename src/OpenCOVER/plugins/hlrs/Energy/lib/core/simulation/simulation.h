#pragma once

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "datastorage.h"
#include "object.h"
#include "scalarproperties.h"

namespace core::simulation {

using ObjectMap = std::map<std::string, std::unique_ptr<Object>>;
using ObjectMapView = std::vector<std::reference_wrapper<ObjectMap>>;

class Simulation {
 public:
  Simulation() = default;

  auto getDataStorage() { return m_dataStorage; }
  const auto &getScalarProperties() const { return m_scalarProperties; }
  auto &getScalarProperties() { return m_scalarProperties; }

  virtual const std::vector<double> *getTimedependentScalar(
      const std::string &species, const std::string &node) const = 0;
  virtual void computeParameters() = 0;

 protected:
  void computeParameter(const ObjectMapView &mapView, float trim = 0.01) {
    std::map<std::string, std::vector<double>> allValues{};

    for (const auto &map : mapView) {
      for (const auto &[_, object] : map.get()) {
        const auto &data = object->getData();
        for (const auto &[key, values] : data)
          allValues[key].insert(allValues[key].end(), values.begin(), values.end());
      }
    }

    for (const auto &[key, values] : allValues) {
      setUnit(key);
      setPreferredColorMap(key);
      computeMinMax(key, values, trim);  // 1% trimming
      computeMaxTimestep(key, values);
      m_scalarProperties.ref()[key].species = key;
    }
  }

  const std::vector<double> *getTimedependentScalar(const ObjectMap &map,
                                                    const std::string &species,
                                                    const std::string &node) const {
    auto it = map.find(node);
    if (it != map.end()) {
      const auto &data = it->second->getData();
      auto dataIt = data.find(species);
      if (dataIt != data.end()) {
        return &dataIt->second;
      } else {
        std::cerr << "Species not found: " << species << std::endl;
      }
    } else {
      std::cerr << "Node not found: " << node << std::endl;
    }
    return nullptr;
  }

  virtual void computeMinMax(const std::string &key,
                             const std::vector<double> &values,
                             const double &trimPercent = 0.00);
  virtual void computeMaxTimestep(const std::string &key,
                                  const std::vector<double> &values);
  virtual void setUnit(const std::string &key);
  virtual void setPreferredColorMap(const std::string &key);

  ScalarProperties m_scalarProperties;
  DataStorage m_dataStorage;
};
}  // namespace core::simulation
