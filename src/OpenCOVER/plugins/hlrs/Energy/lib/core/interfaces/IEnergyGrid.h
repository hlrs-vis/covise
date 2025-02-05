#ifndef _CORE_INTERFACES_IENERGYGRID_H
#define _CORE_INTERFACES_IENERGYGRID_H

#include "IColorable.h"
#include "IDrawables.h"
#include "ITimedependable.h"

namespace core::interface {
class IEnergyGrid : public IDrawables, public IColorable, public ITimedependable {
 public:
  virtual ~IEnergyGrid() = default;
};
}  // namespace core::interface

#endif
