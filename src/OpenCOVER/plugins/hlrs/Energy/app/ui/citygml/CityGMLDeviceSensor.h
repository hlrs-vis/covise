#ifndef _CITYGMLDEVICESENSOR_H
#define _CITYGMLDEVICESENSOR_H

#include <PluginUtil/coSensor.h>
#include <PluginUtil/colors/coColorMap.h>
#include <lib/core/interfaces/IBuilding.h>
#include <lib/core/interfaces/IInfoboard.h>
#include <lib/core/utils/color.h>
#include "app/presentation/TxtInfoboard.h"

#include <memory>
#include <osg/Group>

class CityGMLDeviceSensor : public coPickSensor {
 public:
  CityGMLDeviceSensor(
      osg::ref_ptr<osg::Group> group,
      std::unique_ptr<core::interface::IInfoboard<std::string>> &&infoBoard,
      std::unique_ptr<core::interface::IBuilding> &&drawableBuilding,
      const std::vector<std::string> &textBoxTxt = {});

  ~CityGMLDeviceSensor();
  CityGMLDeviceSensor(const CityGMLDeviceSensor &) = delete;
  CityGMLDeviceSensor &operator=(const CityGMLDeviceSensor &) = delete;

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
  void updateTimestepColors(const std::vector<float> &values,
                            const opencover::ColorMap &map);
  void updateTxtBoxTexts(const std::vector<std::string> &texts);
  void updateTitleOfInfoboard(const std::string &title);
  void setColorMapInShader(const opencover::ColorMap &colorMap);
  void setDataInShader(const std::vector<double> &data, float min, float max);

 private:
  std::unique_ptr<core::interface::IBuilding> m_cityGMLBuilding;
  std::unique_ptr<core::interface::IInfoboard<std::string>> m_infoBoard;
  std::vector<osg::Vec4> m_colors;
  std::vector<std::string> m_textBoxTxt;
  bool m_active = false;
};
#endif
