#include "power.h"
#include "object.h"

namespace core::simulation::power {

void PowerSimulation::init() {
  initScalarProperties(
    {
      std::ref(m_buses),
      std::ref(m_generators),
      std::ref(m_transformators),
      std::ref(m_buildings),
      std::ref(m_cables)
    },
    0.0f
  );
}

const_ScalarVecs PowerSimulation::getTimedependentScalar(
    const std::string &species, const std::string &node) const {
    return ScalarByNameCollector(
      {
          std::ref(m_buses),
          std::ref(m_generators),
          std::ref(m_transformators),
          std::ref(m_buildings),
          std::ref(m_cables)
      },
      node, species).collect();
}

}  // namespace core::simulation::power
