#include "heating.h"

#include <iostream>
namespace core::simulation::heating {

void HeatingSimulation::computeParameters() {
  // TODO: rewrite to use factory pattern
  computeParameter(m_consumers);
  computeParameter(m_producers);
}

const std::vector<double> *HeatingSimulation::getTimedependentScalar(
    const std::string &species, const std::string &node) const {
  if (auto result = Simulation::getTimedependentScalar(m_consumers, species, node))
    return result;
  if (auto result = Simulation::getTimedependentScalar(m_producers, species, node))
    return result;
  std::cerr << "No data found for species: " << species << " in node: " << node
            << "\n";
  return nullptr;
}
}  // namespace core::simulation::heating
