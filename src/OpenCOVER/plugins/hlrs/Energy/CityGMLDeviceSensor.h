#ifndef _CITYGMLDEVICESENSOR_H
#define _CITYGMLDEVICESENSOR_H

#include <core/utils/color.h>
#include <PluginUtil/coSensor.h>
#include <core/interfaces/IBuilding.h>
#include <core/interfaces/IInfoboard.h>

#include <memory>
#include <osg/Group>

class CityGMLDeviceSensor : public coPickSensor {
 typedef core::utils::color::ColorMapExtended ColorMapExtended;
 public:
  CityGMLDeviceSensor(
      osg::ref_ptr<osg::Group> group,
      std::unique_ptr<core::interface::IInfoboard<std::string>> &&infoBoard,
      std::unique_ptr<core::interface::IBuilding> &&drawableBuilding,
      std::shared_ptr<ColorMapExtended> colorMap);

  ~CityGMLDeviceSensor();

  void update() override;
  void setTimestep(int t) { m_cityGMLBuilding->updateTime(t); }
  void activate() override;
  void disactivate() override;
  void updateTime(int timestep);
  void updateColorOfBuilding(const osg::Vec4 &color) {
    m_cityGMLBuilding->updateColor(color);
  }

  auto getDrawables() const { return m_cityGMLBuilding->getDrawables(); }
  osg::Node *getDrawable(size_t index) const {
    return m_cityGMLBuilding->getDrawable(index);
  }
  auto getParent() { return getNode()->asGroup(); }
//   void updateShader();

 private:
  std::unique_ptr<core::interface::IBuilding> m_cityGMLBuilding;
  std::unique_ptr<core::interface::IInfoboard<std::string>> m_infoBoard;
  std::weak_ptr<ColorMapExtended> m_colorMapRef;
  bool m_active = false;
};

#endif
