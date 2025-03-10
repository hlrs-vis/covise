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
  template <typename simulationObject>
  void updateEnergyGridColors(int timestep, std::shared_ptr<EnergyGrid> energyGrid,
                              const ObjectContainer<simulationObject> &container) {
    isDerivedFromObject<simulationObject>();
    for (const auto &[nameOfConsumer, consumer] : container.get()) {
      const auto &name = consumer.getName();
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

  const opencover::ColorMap *m_colorMap = nullptr;
  template <typename simulationObject>
  void computeColors(
      const opencover::ColorMap &color_map,
      const std::map<std::string, simulationObject> &objectContainer) {
    m_colorMap = &color_map;
    isDerivedFromObject<simulationObject>();
    double minKeyVal = 1000.0, maxKeyVal = 1.0;

    try {
      auto simulation = m_simulation.lock();
      if (!simulation) {
        std::cerr << "Simulation is not available for computation of colors."
                  << std::endl;
        return;
      }

      minKeyVal = simulation->getMin(color_map.species());
      maxKeyVal = simulation->getMax(color_map.species());
    } catch (const std::out_of_range &e) {
      std::cerr << "Key not found in minMaxValues: " << color_map.species()
                << std::endl;
      return;
    }

    for (auto &[name, object] : objectContainer) {
      const auto &data = object.getData();
      auto it = data.find(color_map.species());
      if (it == data.end()) {
        std::cerr << "Key not found in data: " << color_map.species() << std::endl;
        continue;
      }
      const auto &values = data.at(color_map.species());
      if (auto color_it = m_colors.find(name); color_it == m_colors.end()) {
        m_colors.insert({name, std::vector<osg::Vec4>(values.size())});
      }
      auto &colors = m_colors[name];

      // color_map
      for (auto i = 0; i < values.size(); ++i) {
        auto interpolated_value = core::utils::math::interpolate(
            values[i], minKeyVal, maxKeyVal, color_map.min(), color_map.max());
        colors[i] = color_map.getColor(interpolated_value);
      }
    }
  }

  template <typename simulationObject>
  void isDerivedFromObject() {
    static_assert(
        std::is_base_of_v<Object, simulationObject>,
        "simulationObject must be derived from core::simulation::heating::Object");
  }
  std::weak_ptr<T> m_parent;  // parent which manages drawable
  std::weak_ptr<Simulation> m_simulation;

 private:
  std::map<std::string, std::vector<osg::Vec4>> m_colors;
};
