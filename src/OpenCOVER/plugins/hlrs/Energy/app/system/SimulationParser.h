#pragma once
#include "Parser.h"
#include "Scenario.h"
#include "app/typedefs.h"

#include <lib/core/simulation/simulationresult.h>

namespace cs = core::simulation;
struct SimulationParser : DataPackageParser<result_ptr>
{
    Scenario scenario;
    SimulationParser(const Scenario &s)
        : scenario(s)
    {
    }
};
