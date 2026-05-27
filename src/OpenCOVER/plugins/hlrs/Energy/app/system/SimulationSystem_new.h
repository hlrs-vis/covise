#pragma once

#include "CityGMLSystem.h"
#include "GridRenderer.h"
#include "GridUIManager.h"
#include "DataLoadManager.h"
#include "DataManager.h"
#include "ScenarioManager.h"
#include "app/ui/SimulationUI.h"
#include <lib/core/ClassLogger.h>

class SimulationSystem final : core::ClassLogger
{
public:
    explicit SimulationSystem(
        GridRenderConfig config,
        core::interface::ui::IComponent *parentMenu,
        const core::interface::ui::IGUIFactory &factory,
        CityGMLSystem *cityGMLSystem,
        osg::ref_ptr<osg::Switch> parent,
        core::interface::ILogger &logger,
        const std::string &scenarioDir);
    void init();
    void enable(bool on);
    void update();
    void updateTime(int timestep);
    bool isEnabled() const { return m_enabled; }

private:
    void initUI();
    void onScenarioSelectionChanged(int scenarioId);

    SimulationUI m_ui;
    DataLoadManager m_dataLoadManager;
    ScenarioManager m_scenarioManager;
    DataManager m_dataManager;
    GridRenderer m_gridRenderer;
    GridUIManager m_gridUIManager;

    bool m_enabled;
    std::string m_scenarioDir;
};
