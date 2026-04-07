#pragma once

#include "FileProvider.h"
#include "DataPackage.h"

class CSVLoader final : public FileProvider
{
public:
    using FileProvider::FileProvider;
    DataPackages load(int scenarioId, EnergyType type) override;
};
