#pragma once
#include <lib/core/interfaces/IInfoboard.h>
#include <lib/core/interfaces/IEnergyGrid.h>

#include "lib/core/simulation/simulationresult.h"
#include "lib/core/simulation/powerresult.h"
#include "osg/OsgImpl.h"

typedef core::interface::IInfoboard<std::string, Drawable> InfoboardImpl;
typedef core::interface::Color Color;
typedef core::interface::Pos Pos;
typedef core::simulation::power::PVData PVData;
typedef OsgBuildingImpl BuildingImpl;
typedef OsgBuildingTimedependImpl BuildingTimedependImpl;
typedef std::shared_ptr<core::interface::IEnergyGrid> grid_ptr;
typedef std::shared_ptr<core::simulation::SimulationResult> result_ptr;
