#include "SimulationSystem_new.h"
#include "EnergyType.h"
#include "CSVLoader.h"
#include "ArrowLoader.h"
#include "Storage.h"
#include <memory>

SimulationSystem::SimulationSystem(
    GridRenderConfig renderConfig,
    core::interface::ui::IComponent *parentMenu,
    const core::interface::ui::IGUIFactory &factory,
    CityGMLSystem *cityGMLSystem,
    osg::ref_ptr<osg::Switch> parent,
    core::interface::ILogger &logger,
    const std::string &scenarioDir)
    : core::ClassLogger(logger, "SimulationSystem")
    , m_ui({ 
        factory, 
        parentMenu, 
        "Simulation", 
        {ENERGYTYPE_RANGE.begin(), ENERGYTYPE_RANGE.end()},
        {FULL_STORAGE_RANGE.begin(), FULL_STORAGE_RANGE.end()}})
    // instead of copying use std::span (view) in c++ 20
    , m_dataLoadManager()
    , m_scenarioManager(scenarioDir)
    , m_dataManager(logger)
    , m_gridRenderer(parent, std::move(renderConfig), logger)
    , m_gridUIManager(m_gridRenderer, parentMenu)
    , m_enabled(true)
    , m_scenarioDir(scenarioDir)
{
}

void SimulationSystem::init()
{
    m_dataLoadManager.registerProvider(Storage::ARROW, std::make_unique<ArrowLoader>(m_scenarioDir));
    m_dataLoadManager.registerProvider(Storage::CSV, std::make_unique<CSVLoader>(m_scenarioDir));

    for (auto type : ENERGYTYPE_RANGE)
        m_gridRenderer.buildGrid(type, m_dataLoadManager);

    initUI();
    m_scenarioManager.setOnScenarioChanged(([this](int id)
    { 
        this->onScenarioSelectionChanged(id); 
    }));
}

void SimulationSystem::initUI()
{
    // Delegate UI interactions to the GridUIManager
    m_ui.setLiftCallback([this](bool on)
        { m_gridUIManager.handleLift(on); });

    m_ui.setOnGridTypeChanged([this](int value)
        {
        if (value >= 0 && value < static_cast<int>(ENERGYTYPE_RANGE.size())) {
            m_gridRenderer.switchTo(ENERGYTYPE_RANGE[value]);
        } else {
            error("Invalid index used for EnergyGridSelection");
        } });

    m_ui.setStorageDefault(EnergyType::HEATING, Storage::CSV);
    m_ui.setStorageDefault(EnergyType::POWER, Storage::ARROW);

    m_ui.setScenarioList(m_scenarioManager.getScenarioNames());

    m_ui.setOnScenarioChanged([this](int id)
    { 
        m_scenarioManager.selectScenario(id);
    });

}

void SimulationSystem::enable(bool on)
{
    m_enabled = on;
    m_gridRenderer.setVisible(on);
}

void SimulationSystem::update()
{
    if (!m_enabled)
        return;
    m_gridRenderer.update();
}

void SimulationSystem::updateTime(int timestep)
{
    if (!m_enabled)
        return;
    m_gridRenderer.updateStep(timestep);
}

void SimulationSystem::onScenarioSelectionChanged(int scenarioId)
{
    if (!m_enabled)
        return;

    auto currentScenario = m_scenarioManager.getSelectedScenario();
    info("Switching to scenario ID: " + std::to_string(scenarioId));

    for (auto type : ENERGYTYPE_RANGE)
    {
        auto storage = m_ui.getSelectedStorage(type);
        m_dataManager.loadScenario(storage, currentScenario, m_dataLoadManager);

        auto result = m_dataManager.getResult(storage, currentScenario, type);
        if (!result)
            continue;

        // TODO: use scalar selector from later UI implementation for species
        std::string species("mass_flow");
        if (type == EnergyType::POWER)
        {
            species = "loading_percent";
            m_gridRenderer.setData(type, result, species);
            // m_gridRenderer.updateColorMapInShader(colorMapMenu.selector->colorMap(),
            //     type);
            m_gridUIManager.updateUIState(type, result);
        }
        // m_gridRenderer.setData(type, result, species);
        // // m_gridRenderer.updateColorMapInShader(colorMapMenu.selector->colorMap(),
        // //     type);
        // m_gridUIManager.updateUIState(type, result);
    }
}
