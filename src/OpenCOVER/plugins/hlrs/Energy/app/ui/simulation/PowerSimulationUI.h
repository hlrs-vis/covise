#pragma once

#include <lib/core/simulation/power.h>

#include "app/presentation/EnergyGrid.h"
#include "app/ui/simulation/BaseSimulationUI.h"

using namespace core::simulation::power;

template <typename T>
class PowerSimulationUI : public BaseSimulationUI<T> {
 public:
  PowerSimulationUI(std::shared_ptr<PowerSimulation> sim, std::shared_ptr<T> parent)
      : BaseSimulationUI<T>(sim, parent) {}
  ~PowerSimulationUI() = default;
  PowerSimulationUI(const PowerSimulationUI&) = delete;
  PowerSimulationUI& operator=(const PowerSimulationUI&) = delete;

  void updateTime(int timestep) override {
    auto parent = this->m_parent.lock();
    if (!parent) return;
    // TODO: rethink this pls => maybe use a visitor pattern
    std::shared_ptr<EnergyGrid> energyGrid =
        std::dynamic_pointer_cast<EnergyGrid>(parent);
    if (energyGrid) {
      auto powerSim = this->powerSimulationPtr();
      if (!powerSim) return;
      auto updateEnergyGridColorsForContainer = [&](auto entities) {
        this->updateEnergyGridColors(timestep, energyGrid, entities);
      };
      updateEnergyGridColorsForContainer(powerSim->Buses());
      updateEnergyGridColorsForContainer(powerSim->Cables());
      updateEnergyGridColorsForContainer(powerSim->Generators());
      updateEnergyGridColorsForContainer(powerSim->Transformators());
    }
  }

  float min(const std::string& species) override {
    return powerSimulationPtr()->getMin(species);
  }

  float max(const std::string& species) override {
    return powerSimulationPtr()->getMax(species);
  }

  void updateTimestepColors(const opencover::ColorMap& map) override {
    // compute colors
    auto powerSim = this->powerSimulationPtr();
    if (!powerSim) return;
    auto computeColorsForContainer = [&](auto container) {
      this->computeColors(map, container);
    };

    computeColorsForContainer(powerSim->Buses().get());
    computeColorsForContainer(powerSim->Cables().get());
    computeColorsForContainer(powerSim->Generators().get());
    computeColorsForContainer(powerSim->Transformators().get());
  }

 private:
  std::shared_ptr<PowerSimulation> powerSimulationPtr() {
    return std::dynamic_pointer_cast<PowerSimulation>(this->m_simulation.lock());
  }
};
