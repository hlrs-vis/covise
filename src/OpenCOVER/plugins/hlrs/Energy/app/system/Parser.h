#pragma once
#include "EnergyType.h"
#include "DataPackage.h"
#include <lib/core/simulation/powerresult.h>
#include <lib/core/simulation/heatingresult.h>
#include <utility>

namespace cs = core::simulation;

struct Parser
{
    virtual std::shared_ptr<cs::SimulationResult> operator()(const CSVDataMap &map) = 0;
    virtual std::shared_ptr<cs::SimulationResult> operator()(const ArrowDataMap &map) = 0;
    virtual std::shared_ptr<cs::SimulationResult> operator()(const ArrowData &data) = 0;
    virtual std::shared_ptr<cs::SimulationResult> operator()(const CSVData &data) = 0;
};

struct PowerParser : public Parser
{
    std::shared_ptr<cs::SimulationResult> operator()(const CSVDataMap &map) override;
    std::shared_ptr<cs::SimulationResult> operator()(const ArrowDataMap &map) override;
    std::shared_ptr<cs::SimulationResult> operator()(const ArrowData &data) override;
    std::shared_ptr<cs::SimulationResult> operator()(const CSVData &data) override;
};

struct HeatingParser : public Parser
{
    std::shared_ptr<cs::SimulationResult> operator()(const CSVDataMap &map) override;
    std::shared_ptr<cs::SimulationResult> operator()(const ArrowDataMap &map) override;
    std::shared_ptr<cs::SimulationResult> operator()(const ArrowData &data) override;
    std::shared_ptr<cs::SimulationResult> operator()(const CSVData &data) override;
};

struct ParseManager
{
    template <typename T>
    std::shared_ptr<cs::SimulationResult> operator()(int scenarioID, EnergyType type, T &&data)
    {
        switch (type)
        {
        case EnergyType::POWER:
            return PowerParser()(std::forward<T>(data));
        case EnergyType::HEATING:
            return HeatingParser()(std::forward<T>(data));
        default:
            return nullptr;
        }
    }
};
