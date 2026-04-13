#include "DataManager.h"
#include "DataFactory.h"

void DataManager::loadScenario(Storage storageType, const Scenario &scenario, DataLoadManager &loader)
{
    for (auto type : ENERGYTYPE_RANGE)
    {
        if (m_cache.count(scenario.id) && m_cache[scenario.id].count(type))
            // check if data is valid/stale
            if (m_cache[scenario.id][type].use_count() > 0 )
                continue;

        auto package = loader.fetch(storageType, scenario, type);
        auto result = DataFactory::create(package, scenario, type);
        m_cache[scenario.id][type] = result;
    }
}
