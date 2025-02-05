#pragma once

#include <string>

#include "simulation.h"

namespace core::simulation::heating {

// Producer and Consumer are derived from Base => probably will differ more in the
// future if heating sim is final
class Producer : public Object {
 public:
  Producer(const std::string &name, const Data &data = {}) : Object(name, data) {}
};

class Consumer : public Object {
 public:
  Consumer(const std::string &name, const Data &data = {}) : Object(name, data) {}
};

class HeatingSimulation : public Simulation {
 public:
  HeatingSimulation() = default;
  auto &Consumers() { return m_consumers; }
  auto &Producers() { return m_producers; }
  const auto &Consumers() const { return m_consumers; }
  const auto &Producers() const { return m_producers; }

  void computeParameters() override;

 private:
  ObjectContainer<Producer> m_consumers;
  ObjectContainer<Consumer> m_producers;
};
}  // namespace core::simulation::heating
