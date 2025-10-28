#pragma once

#include "IColorable.h"
#include "IDrawable.h"
#include "ITimedependable.h"
#include "IUpdateable.h"

namespace core::interface {
class IEnergyGrid : public IDrawable,
                    public IColorable,
                    public ITimedependable,
                    public IUpdateable {
 public:
  ~IEnergyGrid() = default;
};
}  // namespace core::interface
