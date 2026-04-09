#pragma once

#include "CityGMLSystem.h"
#include "GridRenderer.h"
#include "GridUIManager.h"
#include "DataLoadManager.h"
#include "DataManager.h"
#include "Storage.h"
#include "ScenarioManager.h"
#include <lib/core/interfaces/ISystem.h>
#include <lib/core/ClassLogger.h>
#include <cover/coVRPluginSupport.h>

class SimulationSystem final : public core::interface::ISystem, core::ClassLogger {
public:
    SimulationSystem(opencover::coVRPlugin *plugin, opencover::ui::Group *parentMenu,
                   CityGMLSystem *cityGMLSystem, osg::ref_ptr<osg::Switch> parent, core::interface::ILogger &logger, const std::string& scenarioDir);
    void init() override;
    void enable(bool on) override;
    void update() override;
    void updateTime(int timestep) override;
    bool isEnabled() const override { return m_enabled; }
    
private:
    void onScenarioSelectionChanged(int scenarioId);

    DataLoadManager m_dataLoadManager;
    ScenarioManager m_scenarioManager;
    DataManager m_dataManager;
    GridUIManager m_gridUIManager;
    GridRenderer m_gridRenderer;
    //TODO: remove this shit
    opencover::coVRPlugin *m_plugin;
    
    //TODO: put this into GUIManager
    Storage m_currentStorageSelection;
    // int m_currentScenario;
    bool m_enabled;
    std::string m_scenarioDir;
};
