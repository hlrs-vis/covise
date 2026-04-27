#include "SimulationSystem_new.h"
#include "EnergyType.h"
#include "CSVLoader.h"
#include "ArrowLoader.h"
#include "Storage.h"
#include <initializer_list>
#include <memory>

SimulationSystem::SimulationSystem(
    const GridRenderConfig &renderConfig,
    core::interface::ui::IComponent *parentMenu,
    const core::interface::ui::IGUIFactory &factory,
    CityGMLSystem *cityGMLSystem,
    osg::ref_ptr<osg::Switch> parent,
    core::interface::ILogger &logger,
    const std::string &scenarioDir)
    : core::ClassLogger(logger, "SimulationSystem")
    , m_ui(factory, "Simulation", parentMenu)
    , m_dataLoadManager()
    , m_scenarioManager(factory, "ScenarioManager", parentMenu, scenarioDir)
    , m_dataManager(logger)
    , m_gridUIManager(parentMenu)
    , m_gridRenderer(
          parent,
          renderConfig,
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

    constexpr float uplift(30.0f);
    m_ui.setLiftCallback([this, &uplift](bool on)
        {
            auto active = on ? 1 : 0;
            m_gridRenderer.translate(osg::Vec3f(0,0,uplift * active));
        }
    );
    
    m_ui.setEnergyGridSelectionCallback([this](int value)
        {
            if (value > ENERGYTYPE_RANGE.size() || value < 0) {
                error("Invalid index used for EnergyGridSelection");
                return;
            }
            m_gridRenderer.switchTo(ENERGYTYPE_RANGE[value]);
        }
    );

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

    // POWER
    m_dataManager.loadScenario(Storage::ARROW, currentScenario, m_dataLoadManager);

    // HEATING
    m_dataManager.loadScenario(Storage::CSV, currentScenario, m_dataLoadManager);

    std::vector<std::pair<Storage, EnergyType>> StorageEnergyTypeList = { { Storage::ARROW, EnergyType::POWER }, { Storage::CSV, EnergyType::HEATING } };

    for (auto &[storage, type] : StorageEnergyTypeList)
    {
        auto result = m_dataManager.getResult(storage, currentScenario, type);
        if (result)
        {
            // TODO: use scalar selector from later UI implementation for species
            std::string species("mass_flow");
            if (type == EnergyType::POWER)
            {
                species = "loading_percent";
            }
            m_gridRenderer.setData(type, result, species);
            // m_gridRenderer.updateColorMapInShader(colorMapMenu.selector->colorMap(),
            //     type);
            m_gridUIManager.updateUIState(type, result);
        }
    }
}
