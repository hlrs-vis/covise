#pragma once
#include "app/ui/cover/CoverOwner.h"
#include "app/system/CityGMLSystem.h"
#include "app/system/SimulationSystem_new.h"
#include "app/ui/EnergyUI.h"
#include "EnergyLogger.h"

#include <lib/core/interfaces/ui/IGUIFactory.h>
#include <memory>

template <typename T>
constexpr bool IsValidSystem = false;

template <>
constexpr bool IsValidSystem<CityGMLSystem> = true;

template <>
constexpr bool IsValidSystem<SimulationSystem> = true;

class EnergyPlugin : public opencover::coVRPlugin
{
  enum class System { CityGML, Ennovatis, Simulation };

 public:
  EnergyPlugin();
  ~EnergyPlugin();
  EnergyPlugin(const EnergyPlugin &) = delete;
  void operator=(const EnergyPlugin &) = delete;

  bool init() override;
  bool update() override;
  void setTimestep(int t) override;

 private:
  void preFrame() override;

  static constexpr int const getSystemIndex(System system) {
    return static_cast<int>(system);
  }

  template <typename T>
  T *getSystem(System system) {
    if constexpr (IsValidSystem<T>) {
      auto it = m_systems.find(system);
      if (it != m_systems.end()) {
        return dynamic_cast<T *>(it->second.get());
      }
    }
    return nullptr;
  }

  CityGMLSystem *getCityGMLSystem() {
    return getSystem<CityGMLSystem>(System::CityGML);
  }

  SimulationSystem *getSimulationSystem() {
    return getSystem<SimulationSystem>(System::Simulation);
  }

  void initSystems();

  osg::ref_ptr<osg::Switch> m_switch;
  osg::ref_ptr<osg::Switch> m_grid;
  osg::ref_ptr<osg::MatrixTransform> m_Energy;

  std::map<System, std::unique_ptr<core::interface::ISystem>> m_systems;
  std::unique_ptr<core::interface::ui::IGUIFactory> m_factory;

  CoverOwner m_owner;
  EnergyUI m_ui;
  EnergyLogger m_logger;
};
