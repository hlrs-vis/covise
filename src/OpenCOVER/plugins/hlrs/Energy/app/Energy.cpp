#include "Energy.h"

// COVER
#include <cover/coVRPluginSupport.h>

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
}

bool EnergyPlugin::init() {
  initSystems();
  return true;
}

void EnergyPlugin::initSystems() {
  m_systems[System::CityGML] = std::make_unique<CityGMLSystem>(
      this, m_tab, cover->getObjectsRoot(), m_switch);
  m_systems[System::Simulation] =
      std::make_unique<SimulationSystem>(this, m_tab, getCityGMLSystem(), m_grid);

  for (auto &[type, system] : m_systems) {
    system->init();
    system->enable(true);
  }
}
COVERPLUGIN(EnergyPlugin)
