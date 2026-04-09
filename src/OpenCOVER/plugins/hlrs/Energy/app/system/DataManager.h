#pragma once
#include "DataLoadManager.h"
#include "EnergyType.h"
#include "Scenario.h"
#include <lib/core/simulation/simulationresult.h>
#include <map>
#include <memory>

class DataManager
{
public:
    void loadScenario(Storage storageType, const Scenario &scenario, DataLoadManager &loader);
    std::shared_ptr<core::simulation::SimulationResult> getResult(const Scenario &scenario, EnergyType type)
    {
        return m_cache[scenario.id][type];
    }

private:
    // Memory Storage: [ScenarioID][EnergyType] -> Data
    std::map<decltype(Scenario().id), std::map<EnergyType, std::shared_ptr<core::simulation::SimulationResult>>> m_cache;
};
