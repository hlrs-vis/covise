#include "DataManager.h"
#include "DataFactory.h"

void DataManager::loadScenario(Storage storageType, const Scenario &scenario,
    DataLoadManager &loader)
{
    for (auto type : ENERGYTYPE_RANGE)
    {
        if (m_cache.count(storageType) && m_cache[storageType].count(scenario.id) && m_cache[storageType][scenario.id].count(type))
            // check if data is valid/stale
            if (m_cache[storageType][scenario.id][type].use_count() > 0)
                continue;

        auto package = loader.fetch(storageType, scenario, type);
        if (package)
        {
            m_cache[storageType][scenario.id][type] = std::move(DataFactory::create(*package, type, scenario));
        }
        else
        {
            error("Failed to load data for scenario" + scenario.name);
        }
    }
}
