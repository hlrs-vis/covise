#pragma once

#include <string>

#include "object.h"
#include "simulationresult.h"
#include "type.h"

namespace core::simulation::heating {

class HeatingSimulationResult : public SimulationResult {
 public:
  HeatingSimulationResult() = default;
  auto &Consumers() { return m_consumers; }
  auto &Producers() { return m_producers; }
  const auto &Consumers() const { return m_consumers; }
  const auto &Producers() const { return m_producers; }

  void init() override;
  ScalarByNameCollectorResult getTimedependentScalar(
      const std::string &species, const std::string &node) const override;

 private:
  ObjectMap m_consumers;
  ObjectMap m_producers;
};
}  // namespace core::simulation::heating
