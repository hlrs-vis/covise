#pragma once
#include "app/ui/cover/CoverOwner.h"
#include "app/system/CityGMLSystem.h"
#include "app/system/SimulationSystem_new.h"
#include "app/ui/EnergyUI.h"
#include "EnergyLogger.h"

#include <lib/core/interfaces/ui/IGUIFactory.h>
#include <memory>

class EnergyPlugin : public opencover::coVRPlugin
{
 public:
  EnergyPlugin();
  ~EnergyPlugin();
  EnergyPlugin(const EnergyPlugin &) = delete;
  void operator=(const EnergyPlugin &) = delete;

  bool init() override;
  bool update() override;
  void setTimestep(int t) override;

 private:
  void initSystems();

  osg::ref_ptr<osg::Switch> m_switch;
  osg::ref_ptr<osg::Switch> m_grid;
  osg::ref_ptr<osg::MatrixTransform> m_Energy;

  std::unique_ptr<core::interface::ui::IGUIFactory> m_factory;

  CoverOwner m_owner;
  EnergyUI m_ui;
  EnergyLogger m_logger;
  CityGMLSystem m_citygml;
  SimulationSystem m_simulation;
};
