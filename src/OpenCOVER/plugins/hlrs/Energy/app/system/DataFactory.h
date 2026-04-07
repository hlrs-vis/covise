#pragma once
#include "EnergyType.h"
#include "DataPackage.h"
#include <lib/core/simulation/simulationresult.h>
#include <lib/core/simulation/powerresult.h>
#include <lib/core/simulation/heatingresult.h>

namespace cs = core::simulation;

struct SimulationResultVisitor
{
    int scenarioID;
    EnergyType energyType;

    std::shared_ptr<cs::SimulationResult> operator()(CSVData &csvStream) const
    {
        if (!csvStream)
            return nullptr;
        // ... call your parser ...
    }

    std::shared_ptr<cs::SimulationResult> operator()(ArrowData &table) const
    {
        if (!table)
            return nullptr;
        // ... call your parser ...
    }
};

struct SimulationResultMapVisitor
{
    int scenarioID;
    EnergyType energyType;

    std::shared_ptr<cs::SimulationResult> operator()(CSVDataMap &csvStreams) const
    {
        if (!csvStreams.empty())
            return nullptr;
        // ... call your parser ...
    }

    std::shared_ptr<cs::SimulationResult> operator()(ArrowDataMap &tables) const
    {
        if (!tables.empty())
            return nullptr;
        // ... call your parser ...
    }
};

struct DataFactory
{
    static std::shared_ptr<cs::SimulationResult> create(DataPackage &package, int id, EnergyType type)
    {
        return std::visit(SimulationResultVisitor { id, type }, package);
    }

    static std::shared_ptr<cs::SimulationResult> create(DataPackages &packages, int id, EnergyType type)
    {
        return std::visit(SimulationResultMapVisitor { id, type }, packages);
    }
};
