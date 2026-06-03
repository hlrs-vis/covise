#include "heatingresult.h"

namespace core::simulation::heating {

void HeatingSimulationResult::init() {
  initScalarProperties(
    {
      &m_consumers,
      &m_producers
    }
  );
}

ScalarByNameCollectorResult HeatingSimulationResult::getTimedependentScalar(
    const std::string &species, const std::string &node) const {
    ObjectMapView view =
    {
      &m_consumers,
      &m_producers
    };
    return ScalarByNameCollector(view, node, species).collect();
}
}  // namespace core::simulation::heating
