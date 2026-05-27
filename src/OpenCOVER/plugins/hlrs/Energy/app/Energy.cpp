#include "Energy.h"
#include "app/system/CityGMLSystem.h"
#include "app/system/SimulationSystem_new.h"
#include "app/ui/cover/CoverUIFactory.h"
#include <lib/core/constants.h>

// COVER
#include <cover/coVRPluginSupport.h>
#include <memory>

using namespace opencover;

EnergyPlugin::EnergyPlugin()
    : coVRPlugin(COVER_PLUGIN_NAME)
    , m_switch(new osg::Switch())
    , m_grid(new osg::Switch())
    , m_Energy(new osg::MatrixTransform())
    , m_factory(std::make_unique<CoverUIFactory>())
    , m_owner("Energy", cover->ui)
    , m_ui(*m_factory, "EnergyPlugin", &m_owner)
    , m_logger(CONSTANTS::NAMES::LOGGER_NAME)
    , m_citygml(
        this,
        m_ui.getTabMenu(),
        *m_factory,
        cover->getObjectsRoot(),
        m_switch,
        m_logger.getLogger()
    )
    , m_simulation(
        GridRenderConfig{
            this->configFloatArray("General", "offset", std::vector<double> { 0, 0, 0})->value(),
            this->configString("Billboard", "font", "default")->value(),
            {
                { this->configString("General", "projFrom", "default")->value() },
                { this->configString("General", "projTo", "default")->value() } 
            }
        }, 
        m_ui.getTabMenu(),
        *m_factory,
        &m_citygml,
        m_grid,
        m_logger.getLogger(),
        this->configString("Simulation", "scenarioDir", "default")->value()
    )
{
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

bool EnergyPlugin::update()
{
    m_simulation.update();
    m_citygml.update();
    return true;
}

void EnergyPlugin::setTimestep(int t)
{
    m_simulation.updateTime(t);
    m_citygml.updateTime(t);
}

bool EnergyPlugin::init()
{
    m_logger.getLogger().info("Starting Energy Plugin");

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

    initSystems();
    return true;
}

void EnergyPlugin::initSystems()
{
    m_simulation.init();
    m_citygml.init();
    m_simulation.enable(true);
    m_citygml.enable(true);
}
COVERPLUGIN(EnergyPlugin)
