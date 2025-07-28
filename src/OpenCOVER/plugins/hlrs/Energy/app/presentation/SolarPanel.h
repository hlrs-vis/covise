#pragma once
#include <lib/core/interfaces/ISolarPanel.h>

#include <osg/CopyOp>
#include <osg/Geode>
#include <osg/Node>
#include <osgDB/Options>

struct SolarPanelConfig {
  SolarPanelConfig(const std::string &name, const osg::Vec3 &position,
                   osg::ref_ptr<osg::Group> parent, osg::ref_ptr<osg::Node> geo)
      : name(name), position(position), parent(parent), solarPanelNode(geo) {}

  std::string name;
  osg::Vec3 position;
  osg::ref_ptr<osg::Group> parent;
  osg::ref_ptr<osg::Node> solarPanelNode;
};

class SolarPanel : public core::interface::ISolarPanel {
 public:
  //   SolarPanel(const SolarPanelConfig &config);
  SolarPanel(osg::ref_ptr<osg::Node> node) : m_node(node) { init(); }

  ~SolarPanel() {};
  void initDrawables() override;
  void updateDrawables() override;
  void updateColor(const osg::Vec4 &color) override;

 private:
  void init();
  //   SolarPanelConfig m_config;
  osg::ref_ptr<osg::Node> m_node;
};
