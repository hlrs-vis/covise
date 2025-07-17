#pragma once
#include <lib/core/interfaces/ISolarPanel.h>
#include <lib/core/utils/osgUtils.h>

#include <osg/CopyOp>
#include <osg/Geode>
#include <osg/Node>
#include <osgDB/Options>

struct SolarPanelConfig {
  std::string name;
  float zOffset;
  float numMaxPanels;
  float panelWidth;
  float panelHeight;
  osg::Vec4 colorIntensity;
  osg::Matrixd rotation;
  osg::ref_ptr<osg::Group> parent;
  osg::ref_ptr<osg::Geode> geode;
  std::vector<core::utils::osgUtils::instancing::GeometryData> masterGeometryData;
  bool valid() const { return parent && geode && !masterGeometryData.empty(); }
};

class SolarPanel : public core::interface::ISolarPanel {
 public:
  SolarPanel(osg::ref_ptr<osg::Node> node) : m_node(node) { init(); }

  ~SolarPanel() {};
  void initDrawables() override;
  void updateDrawables() override;
  void updateColor(const osg::Vec4 &color) override;

 private:
  void init();
  osg::ref_ptr<osg::Node> m_node;
};
