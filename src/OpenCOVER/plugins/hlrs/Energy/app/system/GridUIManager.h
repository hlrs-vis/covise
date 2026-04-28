#pragma once

#include "EnergyType.h"
#include "app/system/GridRenderer.h"
#include "lib/core/interfaces/ui/IComponent.h"
#include <lib/core/simulation/simulationresult.h>

class GridUIManager
{
public:
    GridUIManager(GridRenderer &renderer, core::interface::ui::IComponent *parentMenu);
    void updateUIState(EnergyType type, std::shared_ptr<core::simulation::SimulationResult> result);
    void handleLift(bool on);

private:
    GridRenderer &m_renderer;
};
