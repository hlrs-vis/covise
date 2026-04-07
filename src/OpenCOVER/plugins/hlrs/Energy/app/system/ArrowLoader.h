#pragma once
#include "FileProvider.h"
#include "DataPackage.h"

class ArrowLoader final : public FileProvider
{
public:
    using FileProvider::FileProvider;
    DataPackages load(int scenarioId, EnergyType type) override;
};
