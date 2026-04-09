#pragma once
#include "EnergyType.h"
#include "Parser.h"
#include <lib/core/simulation/simulationresult.h>
#include <lib/core/simulation/powerresult.h>
#include <lib/core/simulation/heatingresult.h>
#include <memory>

struct ResultVisitor {
    int scenarioID;
    EnergyType energyType;
    
    template<typename T>
    std::shared_ptr<cs::SimulationResult> operator()(T &&data) const
    {
        return ParseManager()(scenarioID, energyType, std::forward<T>(data));
    }
};

struct DataFactory
{
    template<typename T>
    static std::shared_ptr<cs::SimulationResult> create(T &&package, int id, EnergyType type)
    {
        return std::visit(ResultVisitor{ id, type }, std::forward<T>(package));
    }
};
