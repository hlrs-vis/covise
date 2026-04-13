#pragma once
#include "EnergyType.h"
#include "Storage.h"
#include "Provider.h"
#include "DataPackage.h"
#include "Scenario.h"
#include <map>
#include <memory>
#include <optional>

class DataLoadManager
{
public:
    void registerProvider(Storage storageType, std::unique_ptr<Provider<DataPackages>> provider)
    {
        m_provider.insert_or_assign(storageType, std::move(provider));
    }

    std::optional<DataPackages> fetch(Storage storageType, const Scenario &scenario, EnergyType energyType)
    {
        if (auto p_iter = m_provider.find(storageType); p_iter != m_provider.end())
        {
            auto &p = p_iter->second;
            return p->load(scenario, energyType);
        }
        return {};
    }

private:
    std::map<Storage, std::unique_ptr<Provider<DataPackages>>> m_provider;
};
