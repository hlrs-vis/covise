#pragma once

#include "EnergyType.h"
#include "lib/core/interfaces/ui/IComponent.h"
#include <lib/core/simulation/simulationresult.h>

class GridUIManager
{
public:
    GridUIManager(core::interface::ui::IComponent *parentMenu);
    void updateUIState(EnergyType type, std::shared_ptr<core::simulation::SimulationResult> result);
};
