#include "power.h"

namespace core::simulation::power {

void PowerSimulation::computeParameters() {
  // TODO: rewrite to use factorypattern
  computeParameter(m_buses, 0.0f);
  computeParameter(m_generators);
  computeParameter(m_transformators);
  computeParameter(m_buildings, 0.0f);
  computeParameter(m_cables, 0.0f);  // no trimming for cables
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
