#pragma once

#include <PluginUtil/colors/coColorBar.h>
#include <PluginUtil/colors/coColorMap.h>
#include <lib/core/interfaces/IColorable.h>
#include <lib/core/interfaces/IDrawables.h>
#include <lib/core/interfaces/ITimedependable.h>
#include <lib/core/simulation/object.h>
#include <lib/core/simulation/simulation.h>
#include <lib/core/utils/color.h>
#include <lib/core/utils/math.h>

#include <iostream>
#include <memory>

#include "app/presentation/EnergyGrid.h"

using namespace core::simulation;

/**
 * @brief BaseSimulationUI is a template class providing a UI interface for
 * simulation objects.
 *
 * This class is designed to work with simulation objects that are derived from
 * core::interface::IDrawables, core::interface::IColorable, and
 * core::interface::ITimedependable. It manages color mapping and time-dependent
 * updates for simulation visualization.
 *
 * @tparam T Simulation object type, must inherit from IDrawables, IColorable, and
 * ITimedependable.
 *
 * @note Copy construction and assignment are disabled.
 *
 * @param sim Shared pointer to the Simulation instance.
 * @param parent Shared pointer to the parent simulation object.
 *
 * @section Responsibilities
 * - Enforces type constraints on T.
 * - Manages color mapping for simulation objects.
 * - Provides virtual interface for time and color updates.
 * - Computes and updates colors for simulation objects based on simulation data.
 *
 * @section Methods
 * - updateTime(int timestep): Pure virtual, updates simulation state for a given
 * timestep.
 * - min(const std::string &species): Pure virtual, returns minimum value for a
 * species.
 * - max(const std::string &species): Pure virtual, returns maximum value for a
 * species.
 * - updateTimestepColors(const opencover::ColorMap &map): Pure virtual, updates
 * colors for a timestep.
 * - updateEnergyGridColors: Updates colors in an energy grid for a given timestep.
 * - computeColors: Computes color mapping for simulation objects.
 *
 * @section Members
 * - m_colorMap: Pointer to the current color map.
 * - m_parent: Weak pointer to the parent simulation object.
 * - m_simulation: Weak pointer to the simulation instance.
 * - m_colors: Map of object names to their color vectors.
 */
template <typename T>
class BaseSimulationUI {
  static_assert(std::is_base_of_v<core::interface::IDrawables, T>,
                "T must be derived from IDrawable");
  static_assert(std::is_base_of_v<core::interface::IColorable, T>,
                "T must be derived from IColorable");
  static_assert(std::is_base_of_v<core::interface::ITimedependable, T>,
                "T must be derived from ITimeDependable");

 public:
  BaseSimulationUI(std::shared_ptr<Simulation> sim, std::shared_ptr<T> parent)
      : m_simulation(sim), m_parent(parent) {
    if (auto simulation = m_simulation.lock()) simulation->computeParameters();
  }

  ~BaseSimulationUI() = default;
  BaseSimulationUI(const BaseSimulationUI &) = delete;
  BaseSimulationUI &operator=(const BaseSimulationUI &) = delete;

  virtual void updateTime(int timestep) = 0;
  // TODO: make these const
  virtual float min(const std::string &species) = 0;
  virtual float max(const std::string &species) = 0;
  virtual void updateTimestepColors(const opencover::ColorMap &map) = 0;

 protected:
  void updateEnergyGridColors(int timestep, std::shared_ptr<EnergyGrid> energyGrid,
                              const ObjectMapView &objectMapView) {
    for (const auto &objectMap : objectMapView) {
      for (const auto &[nameOfConsumer, consumer] : objectMap.get()) {
        const auto &name = consumer->getName();
        auto colorIt = this->m_colors.find(name);
        if (colorIt == this->m_colors.end()) continue;

        const auto &colors = colorIt->second;
        if (timestep >= colors.size()) continue;

        const auto &color = colors[timestep];
        if (auto point = energyGrid->getPointByName(name)) {
          point->updateColor(color);
        }
      }
    }
  }

  const opencover::ColorMap *m_colorMap = nullptr;
  void computeColors(const opencover::ColorMap &colorMap,
                     const ObjectMapView &objectMapView) {
    m_colorMap = &colorMap;
    double minKeyVal = 1000.0, maxKeyVal = 1.0;

    try {
      auto simulation = m_simulation.lock();
      if (!simulation) {
        std::cerr << "Simulation is not available for computation of colors."
                  << std::endl;
        return;
      }

      minKeyVal = simulation->getMin(colorMap.species());
      maxKeyVal = simulation->getMax(colorMap.species());
    } catch (const std::out_of_range &e) {
      std::cerr << "Key not found in minMaxValues: " << colorMap.species()
                << std::endl;
      return;
    }

    for (const auto &objectMap : objectMapView) {
      // Iterate over each object in the map
      for (auto &[name, object] : objectMap.get()) {
        const auto &data = object->getData();
        auto it = data.find(colorMap.species());
        if (it == data.end()) {
          std::cerr << "Key not found in data: " << colorMap.species() << std::endl;
          continue;
        }
        const auto &values = data.at(colorMap.species());
        if (auto color_it = m_colors.find(name); color_it == m_colors.end()) {
          m_colors.insert({name, std::vector<osg::Vec4>(values.size())});
        }
        auto &colors = m_colors[name];

        // color_map
        for (auto i = 0; i < values.size(); ++i) {
          auto interpolated_value = core::utils::math::interpolate(
              values[i], minKeyVal, maxKeyVal, colorMap.min(), colorMap.max());
          colors[i] = colorMap.getColor(interpolated_value);
        }
      }
    }
  }
  std::weak_ptr<T> m_parent;  // parent which manages drawable
  std::weak_ptr<Simulation> m_simulation;

 private:
  std::map<std::string, std::vector<osg::Vec4>> m_colors;
};
