#include "heatingresult.h"

namespace core::simulation::heating {

void HeatingSimulationResult::init() {
  initScalarProperties(
    {
      std::ref(m_consumers),
      std::ref(m_producers)
    }
  );
}

ScalarByNameCollectorResult HeatingSimulationResult::getTimedependentScalar(
    const std::string &species, const std::string &node) const {
    ObjectMapView view =
    {
      std::ref(m_consumers),
      std::ref(m_producers)
    };
    return ScalarByNameCollector(view, node, species).collect();
}
}  // namespace core::simulation::heating
