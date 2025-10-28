#pragma once
#include <osg/Vec4>

#include "IColorable.h"
#include "IDrawables.h"
#include "ITimedependable.h"

namespace core::interface {
class IBuilding : public IDrawables, public IColorable, public ITimedependable {};
}  // namespace core::interface
