#pragma once

#include "FileProvider.h"
#include "DataPackage.h"
#include "Scenario.h"

class CSVLoader final : public FileProvider
{
public:
    using FileProvider::FileProvider;
    DataPackages load(const Scenario &scenario, EnergyType type) const override;
};
