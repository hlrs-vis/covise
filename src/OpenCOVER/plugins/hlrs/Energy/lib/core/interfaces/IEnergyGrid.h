#pragma once
#include <PluginUtil/colors/coColorMap.h>

#include "../simulation/simulation.h"
#include "IColorable.h"
#include "IDrawables.h"
#include "ITimedependable.h"

namespace core::interface {
class IEnergyGrid : public IDrawables, public IColorable, public ITimedependable {
 public:
  virtual ~IEnergyGrid() = default;
  virtual void update() = 0;
  //   virtual void setColorMap(const opencover::ColorMap& colorMap) = 0;
  //   TODO: remove this later => what a fucking mess
  //   HACK:
  virtual void setColorMap(const opencover::ColorMap& colorMap,
                           const opencover::ColorMap& vm_pu_colormap) = 0;
  virtual void setData(const core::simulation::Simulation& sim,
                       const std::string& species, bool interpolate = false) = 0;
};
}  // namespace core::interface
