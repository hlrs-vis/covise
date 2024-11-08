#ifndef _CITYGMLDEVICESENSOR_H
#define _CITYGMLDEVICESENSOR_H

#include <PluginUtil/coSensor.h>
#include <core/interfaces/IBuilding.h>
#include <core/interfaces/IInfoboard.h>
#include <memory>
#include <osg/Group>

class CityGMLDeviceSensor : public coPickSensor {
public:
  CityGMLDeviceSensor(
      osg::ref_ptr<osg::Group> group,
      std::unique_ptr<core::interface::IInfoboard<std::string>> &&infoBoard,
      std::unique_ptr<core::interface::IBuilding> &&drawableBuilding);

  ~CityGMLDeviceSensor();

  void update() override;
  void setTimestep(int t) { m_cityGMLBuilding->updateTime(t); }
  void activate() override;
  void disactivate() override;
  void updateTime(int timestep);
  void updateColorOfBuilding(const osg::Vec4 &color) {
    m_cityGMLBuilding->updateColor(color);
  }

private:
  std::unique_ptr<core::interface::IBuilding> m_cityGMLBuilding;
  std::unique_ptr<core::interface::IInfoboard<std::string>> m_infoBoard;
  bool m_active = false;
};

#endif
