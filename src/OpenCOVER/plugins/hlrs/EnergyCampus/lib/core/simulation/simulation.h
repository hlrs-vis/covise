#pragma once

#include <string>

#include "collector.h"
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

  virtual const_ScalarVecs getTimedependentScalar(
      const std::string &species, const std::string &node) const = 0;
  virtual void init() = 0;

 protected:
  void initScalarProperties(const ObjectMapView &mapView, float trim = 0.01) {
    auto collector = ScalarMapCollector(mapView);
    auto scalarValues = collector.collect();
    auto processor = ScalarPropertiesProcessor(trim);

    for (const auto &[key, values] : scalarValues)
      processor.init(m_scalarProperties, key, values);
  }

  ScalarProperties m_scalarProperties;
  DataStorage m_dataStorage;
};
}  // namespace core::simulation
