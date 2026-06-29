#pragma once

#include <string>

#include "simulation.h"

namespace core::simulation::heating {

class HeatingSimulation : public Simulation {
 public:
  HeatingSimulation() = default;
  auto &Consumers() { return m_consumers; }
  auto &Producers() { return m_producers; }
  const auto &Consumers() const { return m_consumers; }
  const auto &Producers() const { return m_producers; }

  void computeParameters() override;
  const std::vector<double> *getTimedependentScalar(
      const std::string &species, const std::string &node) const override;

 private:
  ObjectMap m_consumers;
  ObjectMap m_producers;
};
}  // namespace core::simulation::heating
