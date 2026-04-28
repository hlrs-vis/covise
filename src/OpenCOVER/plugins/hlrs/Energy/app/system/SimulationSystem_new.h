#pragma once

#include "CityGMLSystem.h"
#include "GridRenderer.h"
#include "GridUIManager.h"
#include "DataLoadManager.h"
#include "DataManager.h"
#include "ScenarioManager.h"
#include "app/ui/SimulationUI.h"
#include <lib/core/interfaces/ISystem.h>
#include <lib/core/ClassLogger.h>

class SimulationSystem final : public core::interface::ISystem, core::ClassLogger
{
public:
    SimulationSystem(
        const GridRenderConfig &config,
        core::interface::ui::IComponent *parentMenu, 
        const core::interface::ui::IGUIFactory &factory,
        CityGMLSystem *cityGMLSystem, 
        osg::ref_ptr<osg::Switch> parent, 
        core::interface::ILogger &logger, 
        const std::string &scenarioDir);
    void init() override;
    void enable(bool on) override;
    void update() override;
    void updateTime(int timestep) override;
    bool isEnabled() const override { return m_enabled; }

private:
    void onScenarioSelectionChanged(int scenarioId);

    SimulationUI m_ui;
    DataLoadManager m_dataLoadManager;
    ScenarioManager m_scenarioManager;
    DataManager m_dataManager;
    GridUIManager m_gridUIManager;
    GridRenderer m_gridRenderer;

    bool m_enabled;
    std::string m_scenarioDir;
};
