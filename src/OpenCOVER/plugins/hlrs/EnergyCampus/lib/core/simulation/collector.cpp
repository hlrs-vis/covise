#include "collector.h"
#include <iostream>

namespace core::simulation {

ScalarMap ScalarMapCollector::collect() {
  ScalarMap scalarValues;
  for (const auto &map : m_view) {
    const auto &objectMap = map.get();
    for (const auto &[_, object] : objectMap) {
      const auto &objectData = object->getData();
      for (const auto &[key, values] : objectData)
        scalarValues[key].insert(scalarValues[key].end(), values.begin(),
                                 values.end());
    }
  }
  return scalarValues;
}

const_ScalarVecs ScalarByNameCollector::collect() {
  for (const auto &mapRef : m_view) {
    const auto &objectMap = mapRef.get();

    // Find the object
    if (auto it = objectMap.find(m_name); it != objectMap.end()) {
      const auto &data = it->second->getData();

      // Find the specific species within that object
      if (auto dataIt = data.find(m_species); dataIt != data.end()) {
        return &dataIt->second;
      }
      std::cerr << "No data with " << m_species << " available." << "\n";
    }
    std::cerr << "No object with " << m_name << " available." << "\n";
  }
  return nullptr;
}
}  // namespace core::simulation
