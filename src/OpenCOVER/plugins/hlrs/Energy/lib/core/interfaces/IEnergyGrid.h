#pragma once

#include "IColorable.h"
#include "IDrawables.h"
#include "ITimedependable.h"

namespace core::interface {
class IEnergyGrid : public IDrawables, public IColorable, public ITimedependable {
 public:
  virtual ~IEnergyGrid() = default;
  virtual void update() = 0;
};
}  // namespace core::interface
