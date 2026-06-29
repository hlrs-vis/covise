#include "power.h"

namespace core::simulation::power {

void PowerSimulation::computeParameters() {
  computeParameter(
      {std::ref(m_buses), std::ref(m_generators), std::ref(m_transformators),
       std::ref(m_buildings), std::ref(m_cables)},
      0.0f);
}

const std::vector<double> *PowerSimulation::getTimedependentScalar(
    const std::string &species, const std::string &node) const {
  if (auto result = Simulation::getTimedependentScalar(m_buses, species, node))
    return result;
  if (auto result = Simulation::getTimedependentScalar(m_generators, species, node))
    return result;
  if (auto result =
          Simulation::getTimedependentScalar(m_transformators, species, node))
    return result;
  if (auto result = Simulation::getTimedependentScalar(m_buildings, species, node))
    return result;
  if (auto result = Simulation::getTimedependentScalar(m_cables, species, node))
    return result;
  return nullptr;
}

}  // namespace core::simulation::power
