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

const std::vector<double> *PowerSimulation::getTimedependentScalar(
    const std::string &species, const std::string &node) const {
    ObjectMapView view = {
        std::ref(m_buses),
        std::ref(m_generators),
        std::ref(m_transformators),
        std::ref(m_buildings),
        std::ref(m_cables)
    };
    return ScalarByNameCollector(view, node, species).collect();
}

}  // namespace core::simulation::power
