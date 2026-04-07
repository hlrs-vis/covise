#include "SimulationSystem_new.h"
#include "EnergyType.h"
#include "CSVLoader.h"
#include "ArrowLoader.h"
#include "Storage.h"
#include <memory>

SimulationSystem::SimulationSystem(opencover::coVRPlugin *plugin, opencover::ui::Group *parentMenu,
    CityGMLSystem *cityGMLSystem, osg::ref_ptr<osg::Switch> parent, core::interface::ILogger &logger, const std::string& scenarioDir)
    : core::ClassLogger(logger, "SimulationSystem")
    , m_plugin(plugin)
    , m_enabled(false)
    , m_dataManager()
    , m_dataLoadManager()
    , m_scenarioManager(parentMenu)
    , m_gridUIManager(parentMenu)
    , m_gridRenderer(parent)
    , m_currentStorageSelection(Storage::ARROW)
    , m_currentScenario(0)
    , m_scenarioDir(scenarioDir)

{
}

void SimulationSystem::init()
{
    m_dataLoadManager.registerProvider(Storage::ARROW, std::make_unique<ArrowLoader>(m_scenarioDir));
    m_dataLoadManager.registerProvider(Storage::CSV, std::make_unique<CSVLoader>(m_scenarioDir));
    
    m_scenarioManager.setOnScenarioChanged([this](int id){
        this->onScenarioSelectionChanged(id);
    });
}

void SimulationSystem::enable(bool on) {
    
}

void SimulationSystem::update() {
    
}

void SimulationSystem::updateTime(int timestep) {
    
}

void SimulationSystem::onScenarioSelectionChanged(int scenarioId)
{
    m_currentScenario = scenarioId;
    info("Switching to scenario ID: " + std::to_string(scenarioId));

    m_dataManager.loadScenario(m_currentStorageSelection, scenarioId, m_dataLoadManager);

    for (auto type : ENERGYTYPE_RANGE)
    {
        auto result = m_dataManager.getResult(scenarioId, type);
        if (result)
        {
            m_gridRenderer.setData(type, result);
            m_gridUIManager.updateUIState(type, result);
        }
    }
}
