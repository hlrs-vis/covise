#pragma once
#include "IColorable.h"
#include "IDrawables.h"

namespace core::interface {
class ISolarPanel : public IDrawables, public IColorable {};
}  // namespace core::interface
