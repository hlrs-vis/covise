/****************************************************************************\
 **                                                          (C)2024 HLRS  **
 **                                                                        **
 ** Description: OpenCOVER Plug-In for reading building energy data        **
 **                                                                        **
 **                                                                        **
 ** Author: Leyla Kern, Marko Djuric                                       **
 **                                                                        **
 ** History:                                                               **
 **  2024  v1                                                              **
 **  Marko Djuric 02.2024: add ennovatis client                            **
 **  2025  v1.1                                                            **
 **  Marko Djuric 07.2025: refactor EnergyPlugin                           **
 **                                                                        **
\****************************************************************************/
#pragma once
#include "app/CityGMLSystem.h"
#include "app/EnnovatisSystem.h"
#include "app/SimulationSystem.h"

#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/ui/ButtonGroup.h>
#include <cover/ui/CovconfigLink.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Group.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Owner.h>
#include <cover/ui/SelectionList.h>

class EnergyPlugin : public opencover::coVRPlugin,
                     public opencover::ui::Owner,
                     public opencover::coTUIListener {
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
    auto it = m_systems.find(system);
    if (it != m_systems.end()) {
      return dynamic_cast<T *>(it->second.get());
    }
    return nullptr;
  }

  CityGMLSystem *getCityGMLSystem() {
    return getSystem<CityGMLSystem>(System::CityGML);
  }

  EnnovatisSystem *getEnnovatisSystem() {
    return getSystem<EnnovatisSystem>(System::Ennovatis);
  }

  SimulationSystem *getSimulationSystem() {
    return getSystem<SimulationSystem>(System::Simulation);
  }

  void initOverview();
  void initUI();
  void initSystems();

  static EnergyPlugin *m_plugin;
  opencover::ui::Menu *m_tab = nullptr;
  opencover::ui::Menu *m_controlPanel = nullptr;
  opencover::coTUITab *m_tabPanel = nullptr;
  opencover::ui::Button *m_gridControlButton = nullptr;
  opencover::ui::Button *m_energySwitchControlButton = nullptr;

  osg::ref_ptr<osg::Switch> m_switch;
  osg::ref_ptr<osg::Switch> m_grid;
  osg::ref_ptr<osg::MatrixTransform> m_Energy;

  std::map<System, std::unique_ptr<core::interface::ISystem>> m_systems;
};
