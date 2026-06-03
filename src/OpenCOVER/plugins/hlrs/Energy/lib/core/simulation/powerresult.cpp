#include "powerresult.h"
#include "object.h"
// #include <memory>

namespace core::simulation::power {

void PowerSimulationResult::init() {
  initScalarProperties(
    {
      &m_buses,
      &m_generators,
      &m_transformators,
      &m_buildings,
      &m_cables
    },
    0.0f
  );
}

ScalarByNameCollectorResult PowerSimulationResult::getTimedependentScalar(
    const std::string &species, const std::string &node) const {
    return ScalarByNameCollector(
      {
          &m_buses,
          &m_generators,
          &m_transformators,
          &m_buildings,
          &m_cables
      },
      node, species).collect();
}

}  // namespace core::simulation::power
