#pragma once
#include "DataLoadManager.h"
#include "EnergyType.h"
#include <lib/core/simulation/simulationresult.h>
#include <map>
#include <memory>

class DataManager
{
public:
    void loadScenario(Storage storageType, int scenarioId, DataLoadManager &loader);
    std::shared_ptr<core::simulation::SimulationResult> getResult(int scenarioId, EnergyType type)
    {
        return m_cache[scenarioId][type];
    }

private:
    // Memory Storage: [ScenarioID][EnergyType] -> Data
    std::map<int, std::map<EnergyType, std::shared_ptr<core::simulation::SimulationResult>>> m_cache;
};
