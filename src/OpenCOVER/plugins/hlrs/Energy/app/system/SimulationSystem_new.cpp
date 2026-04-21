#include "SimulationSystem_new.h"
#include "EnergyType.h"
#include "CSVLoader.h"
#include "ArrowLoader.h"
#include "Storage.h"
#include <memory>

SimulationSystem::SimulationSystem(opencover::coVRPlugin *plugin, opencover::ui::Group *parentMenu,
    CityGMLSystem *cityGMLSystem, osg::ref_ptr<osg::Switch> parent, core::interface::ILogger &logger, const std::string &scenarioDir)
    : core::ClassLogger(logger, "SimulationSystem")
    , m_plugin(plugin)
    , m_dataLoadManager()
    , m_scenarioManager(parentMenu, scenarioDir)
    , m_dataManager(logger)
    , m_gridUIManager(parentMenu)
    , m_gridRenderer(
          parent,
          { { plugin->configFloatArray("General", "offset", std::vector<double> { 0, 0, 0 })->value() },
              { plugin->configString("Billboard", "font", "default")->value() },
              { m_plugin->configString("General", "projFrom", "default")->value() },
              { m_plugin->configString("General", "projTo", "default")->value() }
          },
          logger)
    , m_currentStorageSelection(Storage::ARROW)
    , m_enabled(false)
    , m_scenarioDir(scenarioDir)
{
}

void SimulationSystem::init()
{
    m_dataLoadManager.registerProvider(Storage::ARROW, std::make_unique<ArrowLoader>(m_scenarioDir));
    m_dataLoadManager.registerProvider(Storage::CSV, std::make_unique<CSVLoader>(m_scenarioDir));

    for (auto type : ENERGYTYPE_RANGE)
        m_gridRenderer.buildGrid(type, m_dataLoadManager);

    m_scenarioManager.setOnScenarioChanged([this](int id)
        { this->onScenarioSelectionChanged(id); });
}

void SimulationSystem::enable(bool on)
{
    m_gridRenderer.setVisible(on);
}

void SimulationSystem::update()
{
    m_gridRenderer.update();
}

void SimulationSystem::updateTime(int timestep)
{
    m_gridRenderer.updateStep(timestep);
}

void SimulationSystem::onScenarioSelectionChanged(int scenarioId)
{
    auto currentScenario = m_scenarioManager.getScenario();
    info("Switching to scenario ID: " + std::to_string(scenarioId));

    m_dataManager.loadScenario(m_currentStorageSelection, currentScenario, m_dataLoadManager);

    for (auto type : ENERGYTYPE_RANGE)
    {
        auto result = m_dataManager.getResult(currentScenario, type);
        if (result)
        {
            // TODO: use scalar selector from later UI implementation for species
            if (type == EnergyType::POWER)
            {
                // m_gridRenderer.updateColorMapInShader(colorMapMenu.selector->colorMap(),
                //     type);
                m_gridRenderer.setData(type, result, "loading_percent");
            }
            m_gridUIManager.updateUIState(type, result);
        }
    }
}
