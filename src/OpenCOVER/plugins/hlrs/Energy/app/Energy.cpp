#include "Energy.h"
#include "lib/core/constants.h"

// COVER
#include <cover/coVRPluginSupport.h>

using namespace opencover;

EnergyPlugin::EnergyPlugin()
    : coVRPlugin(COVER_PLUGIN_NAME)
    , m_ui("EnergyPlugin", cover->ui)
    , m_switch(new osg::Switch())
    , m_grid(new osg::Switch())
    , m_Energy(new osg::MatrixTransform())
    , m_logger(CONSTANTS::NAMES::LOGGER_NAME)
{
    m_logger.info("Starting Energy Plugin");

    // need to save the config on exit => will only be saved when COVER is closed
    // correctly via q or closing the window
    
    config()->setSaveOnExit(true);

    m_Energy->setName("Energy");
    cover->getObjectsRoot()->addChild(m_Energy);

    m_switch->setName("Switch");

    m_grid->setName("EnergyGrids");

    m_Energy->addChild(m_switch);
    m_Energy->addChild(m_grid);

    m_ui.setSwitchCallback(
        [this](bool value)
        {
            if (value)
            {
                m_Energy->addChild(m_switch);
            }
            else
            {
                m_Energy->removeChild(m_switch);
            }
        });
    m_ui.setControlCallback(
        [this](bool value)
        {
    if (value) {
      m_Energy->addChild(m_grid);
    } else {
      m_Energy->removeChild(m_grid);
        
    } });
}

EnergyPlugin::~EnergyPlugin()
{
    auto root = cover->getObjectsRoot();

    if (m_Energy)
    {
        root->removeChild(m_Energy.get());
    }

    config()->save();
}

void EnergyPlugin::preFrame()
{
    auto simSystem = getSimulationSystem();
    if (!simSystem)
        simSystem->preFrame();
}

bool EnergyPlugin::update()
{
    for (auto &[type, system] : m_systems)
        if (system)
            system->update();

    return false;
}

void EnergyPlugin::setTimestep(int t)
{
    for (auto &[type, system] : m_systems)
        if (system)
            system->updateTime(t);
}

bool EnergyPlugin::init()
{
    initSystems();
    return true;
}

void EnergyPlugin::initSystems()
{
    m_systems[System::CityGML] = std::make_unique<CityGMLSystem>(
        this, m_ui.getTabMenu(), cover->getObjectsRoot(), m_switch, m_logger);
    m_systems[System::Simulation] = std::make_unique<SimulationSystem>(this, m_ui.getTabMenu(), getCityGMLSystem(), m_grid);

    for (auto &[type, system] : m_systems)
    {
        system->init();
        system->enable(true);
    }
}
COVERPLUGIN(EnergyPlugin)
