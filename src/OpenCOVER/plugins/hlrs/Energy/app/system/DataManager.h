#pragma once
#include "DataLoadManager.h"
#include "EnergyType.h"
#include "Scenario.h"
#include <lib/core/simulation/simulationresult.h>
#include <lib/core/Logger.h>
#include <map>
#include <memory>

class DataManager
{
public:
    DataManager(Logger logger)
        : m_logger(std::move(logger))
    {
        m_logger.setPrefix("DataManager");
    }

    void loadScenario(Storage storageType, const Scenario &scenario, DataLoadManager &loader);
    std::shared_ptr<core::simulation::SimulationResult> getResult(Storage storageType, const Scenario &scenario, EnergyType type)
    {
        return m_cache[storageType][scenario.id][type];
    }

private:
    // Memory Storage: [Storage][ScenarioID][EnergyType] -> Data
    std::map<Storage, std::map<decltype(Scenario().id), std::map<EnergyType, std::shared_ptr<core::simulation::SimulationResult>>>> m_cache;
    Logger m_logger;
};
