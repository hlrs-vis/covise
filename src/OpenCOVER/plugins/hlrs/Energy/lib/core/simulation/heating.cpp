#include "heating.h"

#include <iostream>
namespace core::simulation::heating {

void HeatingSimulation::computeParameters() {
  computeParameter({std::ref(m_consumers), std::ref(m_producers)});
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
