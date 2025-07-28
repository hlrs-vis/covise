/****************************************************************************\
**                                                          (C)2024 HLRS  **
**                                                                        **
** Description: OpenCOVER Plug-In for reading building energy data        **
**                                                                        **
**                                                                        **
** Author: Leyla Kern, Marko Djuric                                       **
**                                                                        **
** TODO:                                                                  **
**  [ ] fetch lat lon from googlemaps                                     **
**  [x] make REST lib independent from ennovatis general use              **
**  [x] update via REST in background                                     **
**                                                                        **
** History:                                                               **
**  2024  v1                                                              **
**  Marko Djuric 02.2024: add ennovatis client                            **
**  Marko Djuric 10.2024: add citygml interface                           **
**                                                                        **
**  TODO: mapdrap for z coord of EnergyGrids                              **
**                                                                        **
\****************************************************************************/

#include "Energy.h"

// COVER
#include <cover/coVRPluginSupport.h>
#include <gdal_priv.h>

using namespace opencover;

EnergyPlugin *EnergyPlugin::m_plugin = nullptr;

EnergyPlugin::EnergyPlugin()
    : coVRPlugin(COVER_PLUGIN_NAME),
      ui::Owner("EnergyPlugin", cover->ui),
      m_switch(new osg::Switch()),
      m_grid(new osg::Switch()),
      m_Energy(new osg::MatrixTransform()) {
  // need to save the config on exit => will only be saved when COVER is closed
  // correctly via q or closing the window
  config()->setSaveOnExit(true);

  fprintf(stderr, "Starting Energy Plugin\n");
  m_plugin = this;

  m_Energy->setName("Energy");
  cover->getObjectsRoot()->addChild(m_Energy);

  m_switch->setName("Switch");

  m_grid->setName("EnergyGrids");

  m_Energy->addChild(m_switch);
  m_Energy->addChild(m_grid);

  GDALAllRegister();

  initUI();
}

EnergyPlugin::~EnergyPlugin() {
  auto root = cover->getObjectsRoot();

  if (m_Energy) {
    root->removeChild(m_Energy.get());
  }

  config()->save();
  m_plugin = nullptr;
}

void EnergyPlugin::initUI() {
  m_tab = new ui::Menu("Energy_Campus", m_plugin);
  m_tab->setText("Energy Campus");

  initOverview();
}

void EnergyPlugin::initOverview() {
  m_controlPanel = new ui::Menu(m_tab, "Control");
  m_controlPanel->setText("Control_Panel");

  m_energySwitchControlButton = new ui::Button(m_controlPanel, "BuildingSwitch");
  m_energySwitchControlButton->setText("Buildings");
  m_energySwitchControlButton->setCallback([this](bool value) {
    if (value) {
      m_Energy->addChild(m_switch);
    } else {
      m_Energy->removeChild(m_switch);
    }
  });
  m_energySwitchControlButton->setState(true);

  m_gridControlButton = new ui::Button(m_controlPanel, "GridSwitch");
  m_gridControlButton->setText("Grid");
  m_gridControlButton->setCallback([this](bool value) {
    if (value) {
      m_Energy->addChild(m_grid);
    } else {
      m_Energy->removeChild(m_grid);
    }
  });
  m_gridControlButton->setState(true);
}

void EnergyPlugin::preFrame() {
  auto simSystem = getSimulationSystem();
  if (!simSystem) simSystem->preFrame();
}

bool EnergyPlugin::update() {
  for (auto &[type, system] : m_systems)
    if (system) system->update();

  return false;
}

void EnergyPlugin::setTimestep(int t) {
  for (auto &[type, system] : m_systems)
    if (system) system->updateTime(t);

  // this is a workaround for the fact that the energy grids are added in the same
  // order as they appear in the the constructor

  //   auto &energyGrid = m_energyGrids[m_energygridBtnGroup->value()];
  //   if (energyGrid.grid) energyGrid.grid->updateTime(t);
}

bool EnergyPlugin::init() {
  initSystems();
  return true;
}

void EnergyPlugin::initSystems() {
  m_systems[System::Ennovatis] =
      std::make_unique<EnnovatisSystem>(this, m_tab, m_switch);
  m_systems[System::CityGML] = std::make_unique<CityGMLSystem>(
      this, m_tab, cover->getObjectsRoot(), m_switch);
  m_systems[System::Simulation] =
      std::make_unique<SimulationSystem>(this, m_tab, m_grid);

  for (auto &[type, system] : m_systems) {
    system->init();
    system->enable(true);
  }
}
COVERPLUGIN(EnergyPlugin)
