#pragma once
#include "EnergyType.h"
#include "Storage.h"
#include "Provider.h"
#include "DataPackage.h"
#include "Scenario.h"
#include <map>
#include <memory>
#include <stdexcept>

class DataLoadManager
{
public:
    void registerProvider(Storage type, std::unique_ptr<Provider<DataPackages>> provider)
    {
        m_provider[type] = std::move(provider);
    }

    auto fetch(Storage storageType, const Scenario &scenario, EnergyType energyType)
    {
        loaderExist(storageType);
        return m_provider[storageType]->load(scenario, energyType);
    }

private:
    void loaderExist(Storage storageType) {
        if (m_provider.find(storageType) == m_provider.end())
            throw std::runtime_error("No loader for this storage type!");
    }

    std::map<Storage, std::unique_ptr<Provider<DataPackages>>> m_provider;
};
