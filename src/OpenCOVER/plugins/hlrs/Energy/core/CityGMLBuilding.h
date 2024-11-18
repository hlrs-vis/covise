#ifndef _CORE_CITYGMLBUILDING_H
#define _CORE_CITYGMLBUILDING_H

#include "interfaces/IBuilding.h"
#include "utils/osgUtils.h"

namespace core {

class CityGMLBuilding : public interface::IBuilding {
 public:
  CityGMLBuilding(const utils::osgUtils::Geodes &geodes);
  void initDrawables() override;
  void updateColor(const osg::Vec4 &color) override;
  void updateTime(int timestep) override;
  void updateDrawables() override;
  std::unique_ptr<osg::Vec4> getColorInRange(float value, float maxValue) override;
};
}  // namespace core
#endif
