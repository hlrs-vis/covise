#pragma once

#include <cover/ui/Group.h>
#include "EnergyType.h"
#include <lib/core/simulation/simulationresult.h>

class GridUIManager
{
public:
    GridUIManager(opencover::ui::Group *parentMenu);
    void updateUIState(EnergyType type, std::shared_ptr<core::simulation::SimulationResult> result);
};
