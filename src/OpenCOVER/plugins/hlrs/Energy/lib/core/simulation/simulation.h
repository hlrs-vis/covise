#pragma once

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "datastorage.h"
#include "object.h"
#include "scalarproperties.h"
#include "scalarpropertiesprocessor.h"

namespace core::simulation {

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
  void initScalarProperties(const ObjectMapView &mapView, float trim = 0.01) {
    auto scalarValues = collectScalarValues(mapView);
    auto processor = ScalarPropertiesProcessor(trim);

    for (const auto &[key, values] : scalarValues)
      processor.init(m_scalarProperties, key, values);
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

  ScalarProperties m_scalarProperties;
  DataStorage m_dataStorage;

 private:
  std::map<std::string, std::vector<double>> collectScalarValues(
      const ObjectMapView &mapView) {
    std::map<std::string, std::vector<double>> scalarValues{};

    for (const auto &map : mapView) {
      const auto &objectMap = map.get();
      for (const auto &[_, object] : objectMap) {
        const auto &objectData = object->getData();
        for (const auto &[key, values] : objectData)
          scalarValues[key].insert(
            scalarValues[key].end(),
            values.begin(),
            values.end()
          );
      }
    }
    return scalarValues;
  }
};
}  // namespace core::simulation
