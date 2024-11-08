#ifndef _CORE_CITYGMLBUILDING_H
#define _CORE_CITYGMLBUILDING_H

#include "interfaces/IBuilding.h"

namespace core {

class CityGMLBuilding : public interface::IBuilding {
public:
  CityGMLBuilding(osg::Geode *geode);
  void initDrawable() override;
  void updateColor(const osg::Vec4 &color) override;
  void updateTime(int timestep) override;
  void updateDrawable() override;
  std::unique_ptr<osg::Vec4> getColorInRange(float value,
                                             float maxValue) override;
};
} // namespace core
#endif
