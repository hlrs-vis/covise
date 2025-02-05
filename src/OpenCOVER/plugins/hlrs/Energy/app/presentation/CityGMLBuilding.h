#ifndef _CORE_CITYGMLBUILDING_H
#define _CORE_CITYGMLBUILDING_H

#include <lib/core/interfaces/IBuilding.h>
#include <lib/core/utils/osgUtils.h>

#include <memory>

class CityGMLBuilding : public core::interface::IBuilding {
 public:
  CityGMLBuilding(const core::utils::osgUtils::Geodes &geodes);
  void initDrawables() override;
  void updateColor(const osg::Vec4 &color) override;
  void updateTime(int timestep) override;
  void updateDrawables() override;
  std::unique_ptr<osg::Vec4> getColorInRange(float value, float maxValue) override;
};
#endif
