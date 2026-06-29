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

/**
 * @class SolarPanel
 * @brief Represents a solar panel and provides functionality to manage its graphical representation.
 *
 * Inherits from core::interface::ISolarPanel and encapsulates an OSG node representing the solar panel.
 * Provides methods to initialize, update, and modify the appearance of the solar panel's drawables.
 *
 * @note The class is intended for use within the OpenCOVER Energy plugin presentation layer.
 *
 * @see core::interface::ISolarPanel
 */
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
