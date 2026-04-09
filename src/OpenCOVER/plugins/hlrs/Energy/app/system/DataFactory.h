#pragma once
#include "EnergyType.h"
#include "Parser.h"
#include "Scenario.h"
#include <lib/core/simulation/simulationresult.h>
#include <lib/core/simulation/powerresult.h>
#include <lib/core/simulation/heatingresult.h>
#include <memory>

struct ResultVisitor {
    Scenario scenario;
    EnergyType energyType;
    
    template<typename T>
    std::shared_ptr<cs::SimulationResult> operator()(T &&data) const
    {
        return ParseManager()(scenario, energyType, std::forward<T>(data));
    }
};

struct DataFactory
{
    template<typename T>
    static std::shared_ptr<cs::SimulationResult> create(T &&package, const Scenario &scenario, EnergyType type)
    {
        return std::visit(ResultVisitor{ scenario, type }, std::forward<T>(package));
    }
};
