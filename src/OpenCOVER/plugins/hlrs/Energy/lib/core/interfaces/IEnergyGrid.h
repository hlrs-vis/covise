#pragma once

#include "IColorable.h"
#include "IDrawables.h"
#include "ITimedependable.h"
#include "IUpdateable.h"

namespace core::interface {
class IEnergyGrid : public IDrawables,
                    public IColorable,
                    public ITimedependable,
                    public IUpdateable {
 public:
  ~IEnergyGrid() = default;
};
}  // namespace core::interface
