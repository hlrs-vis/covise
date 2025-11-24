#include "collector.h"

namespace core::simulation {

ScalarValuesMap ScalarCollector::collect() {
  ScalarValuesMap scalarValues;
  for (const auto &map : m_view) {
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
}  // namespace core::simulation
