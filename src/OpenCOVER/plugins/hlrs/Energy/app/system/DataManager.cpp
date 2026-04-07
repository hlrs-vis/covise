#include "DataManager.h"
#include "DataFactory.h"

void DataManager::loadScenario(Storage storageType, int scenarioId, DataLoadManager &loader)
{
    for (auto type : ENERGYTYPE_RANGE)
    {
        if (m_cache.count(scenarioId) && m_cache[scenarioId].count(type)) 
            continue;

        auto package = loader.fetch(storageType, scenarioId, type);
        auto result = DataFactory::create(package, scenarioId, type);
        m_cache[scenarioId][type] = result;
    }
}
