#pragma once

#include <lib/core/simulation/power.h>

#include "app/ui/simulation/BaseSimulationUI.h"
#include "app/presentation/EnergyGrid.h"

using namespace core::simulation::power;

template <typename T>
class PowerSimulationUI : public BaseSimulationUI<T> {
 public:
  PowerSimulationUI(std::shared_ptr<PowerSimulation> sim,
                      std::shared_ptr<T> parent, std::shared_ptr<ColorMap> colorMap)
      : BaseSimulationUI<T>(sim, parent, colorMap) {}
  ~PowerSimulationUI() = default;
  PowerSimulationUI(const PowerSimulationUI &) = delete;
  PowerSimulationUI &operator=(const PowerSimulationUI &) = delete;

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
      updateEnergyGridColorsForContainer(powerSim->Generators());
      updateEnergyGridColorsForContainer(powerSim->Transformators());
    }
  }

  void updateTimestepColors(const std::string &key, float min = 0.0f,
                            float max = 1.0f, bool resetMinMax = false) override {
    auto color_map = this->m_colorMapRef.lock();
    if (!color_map) {
      std::cerr << "ColorMap is not available for update of colors." << std::endl;
      return;
    }

    if (min > max) min = max;
    color_map->max = max;
    color_map->min = min;

    if (resetMinMax) {
      auto &[res_min, res_max] = powerSimulationPtr()->getMinMax(key);
      color_map->max = res_max;
      color_map->min = res_min;
    }

    // compute colors
    auto powerSim = this->powerSimulationPtr();
    if (!powerSim) return;
    auto computeColorsForContainer = [&](auto container) {
      this->computeColors(color_map, key, min, max, container);
    };

    computeColorsForContainer(powerSim->Buses().get());
    computeColorsForContainer(powerSim->Generators().get());
    computeColorsForContainer(powerSim->Transformators().get());
  }

 private:
  std::shared_ptr<PowerSimulation> powerSimulationPtr() {
    return std::dynamic_pointer_cast<PowerSimulation>(this->m_simulation.lock());
  }
};
