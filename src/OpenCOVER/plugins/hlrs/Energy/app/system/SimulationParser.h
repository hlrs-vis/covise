#pragma once
#include "Parser.h"
#include "Scenario.h"

#include <lib/core/simulation/simulationresult.h>

namespace cs = core::simulation;
using result_ptr = std::shared_ptr<cs::SimulationResult>;

struct SimulationParser : DataPackageParser<result_ptr>
{
    Scenario scenario;
    SimulationParser(const Scenario &s)
        : scenario(s)
    {
    }
};
