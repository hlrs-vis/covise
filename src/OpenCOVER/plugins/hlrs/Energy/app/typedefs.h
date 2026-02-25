#pragma once
#include <lib/core/interfaces/IInfoboard.h>

#include "OsgImpl.h"

typedef core::interface::IInfoboard<std::string, Drawable> InfoboardImpl;
typedef core::interface::Color Color;
typedef OsgBuildingImpl BuildingImpl;
typedef OsgBuildingTimedependImpl BuildingTimedependImpl;
