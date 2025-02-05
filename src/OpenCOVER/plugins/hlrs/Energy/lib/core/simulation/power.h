#pragma once
#include <string>

#include "simulation.h"

namespace core::simulation::power {

class Bus : public Object {
 public:
  Bus(const std::string &name, const Data &data = {}) : Object(name, data) {}
};

class Generator : public Object {
 public:
  Generator(const std::string &name, const Data &data = {}) : Object(name, data) {}
};

class Transformator : public Object {
 public:
  Transformator(const std::string &name, const Data &data = {})
      : Object(name, data) {}
};

class PowerSimulation : public Simulation {
 public:
  PowerSimulation() = default;

  void computeParameters() override;
  auto &Buses() { return m_buses; }
  auto &Generators() { return m_generators; }
  auto &Transformators() { return m_transformators; }
  const auto &Buses() const { return m_buses; }
  const auto &Generators() const { return m_generators; }
  const auto &Transformators() const { return m_transformators; }

 private:
  ObjectContainer<Bus> m_buses;
  ObjectContainer<Generator> m_generators;
  ObjectContainer<Transformator> m_transformators;
};

}  // namespace core::simulation::power
