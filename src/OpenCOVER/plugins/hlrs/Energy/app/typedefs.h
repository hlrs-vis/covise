#pragma once
#include <lib/core/interfaces/IInfoboard.h>

#include "osg/OsgImpl.h"

typedef core::interface::IInfoboard<std::string, Drawable> InfoboardImpl;
typedef core::interface::Color Color;
typedef core::interface::Pos Pos;
typedef OsgBuildingImpl BuildingImpl;
typedef OsgBuildingTimedependImpl BuildingTimedependImpl;
