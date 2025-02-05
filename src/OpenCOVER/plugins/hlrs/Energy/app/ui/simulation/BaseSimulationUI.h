#pragma once

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

typedef covise::ColorMap ColorMap;

template <typename T>
class BaseSimulationUI {
  static_assert(std::is_base_of_v<core::interface::IDrawables, T>,
                "T must be derived from IDrawable");
  static_assert(std::is_base_of_v<core::interface::IColorable, T>,
                "T must be derived from IColorable");
  static_assert(std::is_base_of_v<core::interface::ITimedependable, T>,
                "T must be derived from ITimeDependable");

 public:
  BaseSimulationUI(std::shared_ptr<Simulation> sim, std::shared_ptr<T> parent,
                   std::shared_ptr<ColorMap> colorMap)
      : m_simulation(sim), m_parent(parent), m_colorMapRef(colorMap) {
    if (auto simulation = m_simulation.lock()) simulation->computeParameters();
  }

  ~BaseSimulationUI() = default;
  BaseSimulationUI(const BaseSimulationUI &) = delete;
  BaseSimulationUI &operator=(const BaseSimulationUI &) = delete;

  virtual void updateTime(int timestep) = 0;
  virtual void updateTimestepColors(const std::string &key, float min = 0.0f,
                                    float max = 1.0f, bool resetMinMax = false) = 0;

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

  template <typename simulationObject>
  void computeColors(
      std::shared_ptr<ColorMap> color_map, const std::string &key, float min,
      float max, const std::map<std::string, simulationObject> &objectContainer) {
    isDerivedFromObject<simulationObject>();
    double minKeyVal = 0.0, maxKeyVal = 1.0;

    try {
      auto simulation = m_simulation.lock();
      if (!simulation) {
        std::cerr << "Simulation is not available for computation of colors."
                  << std::endl;
        return;
      }

      auto &[min_val, max_val] = simulation->getMinMax(key);
      minKeyVal = min_val;
      maxKeyVal = max_val;
    } catch (const std::out_of_range &e) {
      std::cerr << "Key not found in minMaxValues: " << key << std::endl;
      return;
    }

    for (auto &[name, object] : objectContainer) {
      const auto &data = object.getData();
      const auto &values = data.at(key);
      if (auto color_it = m_colors.find(name); color_it == m_colors.end()) {
        m_colors.insert({name, std::vector<osg::Vec4>(values.size())});
      }
      auto &colors = m_colors[name];

      // color_map
      for (auto i = 0; i < values.size(); ++i) {
        auto interpolated_value = core::utils::math::interpolate(
            values[i], minKeyVal, maxKeyVal, min, max);
        auto color = covise::getColor(interpolated_value, *color_map, min, max);
        colors[i] = color;
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
  std::weak_ptr<ColorMap> m_colorMapRef;
  std::weak_ptr<Simulation> m_simulation;
  std::map<std::string, std::vector<osg::Vec4>> m_colors;
};
