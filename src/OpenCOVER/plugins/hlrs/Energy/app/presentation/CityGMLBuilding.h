#ifndef _CORE_CITYGMLBUILDING_H
#define _CORE_CITYGMLBUILDING_H

// #include <lib/core/simulation/simulation.h>
#include <lib/core/interfaces/IBuilding.h>
#include <lib/core/utils/osgUtils.h>
#include <PluginUtil/coShaderUtil.h>

#include <memory>

class CityGMLBuilding : public core::interface::IBuilding {
 public:
  CityGMLBuilding(const core::utils::osgUtils::Geodes &geodes);
  void initDrawables() override;
  void updateColor(const osg::Vec4 &color) override;
  void updateTime(int timestep) override;
  void updateDrawables() override;
  std::unique_ptr<osg::Vec4> getColorInRange(float value, float maxValue) override;

  void setColorMapInShader(const opencover::ColorMap &colorMap);
  void setDataInShader(const std::vector<double> &data, float min, float max);
  bool hasShader() const { return !m_shaders.empty(); }

private:
  std::vector<opencover::coVRShader *> m_shaders;
};
#endif
