#pragma once
#include <string>

#include "simulation.h"

namespace core::simulation::power {

struct PVData {
  std::string cityGMLID;
  float energyYearlyKWhMax{0};
  float pvAreaQm{0};
  float co2savings{0};
  float area{0};
  int numPanelsMax{0};
};

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

class Cable : public Object {
 public:
  Cable(const std::string &name, const Data &data = {}) : Object(name, data) {}
};

class Building : public Object {
 public:
  Building(const std::string &name, const Data &data = {}) : Object(name, data) {}
};

class PowerSimulation : public Simulation {
 public:
  PowerSimulation() = default;

  void computeParameters() override;
  auto &Buses() { return m_buses; }
  auto &Generators() { return m_generators; }
  auto &Transformators() { return m_transformators; }
  auto &Cables() { return m_cables; }
  auto &Buildings() { return m_buildings; }
  const auto &Buses() const { return m_buses; }
  const auto &Generators() const { return m_generators; }
  const auto &Transformators() const { return m_transformators; }
  const auto &Cables() const { return m_cables; }
  const auto &Buildings() const { return m_buildings; }
  const std::vector<double> *getTimedependentScalar(
      const std::string &species, const std::string &node) const override;

 private:
  ObjectContainer<Object> m_buses;
  ObjectContainer<Object> m_generators;
  ObjectContainer<Object> m_transformators;
  ObjectContainer<Object> m_cables;
  ObjectContainer<Object> m_buildings;
};

}  // namespace core::simulation::power
