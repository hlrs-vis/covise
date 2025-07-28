#pragma once
#include <lib/core/interfaces/IBuilding.h>
#include <lib/core/utils/color.h>

#include <memory>
#include <osg/Vec3>
#include <osg/Vec4>

struct CylinderAttributes {
  typedef core::utils::color::ColorMap ColorMap;
  CylinderAttributes(const float &rad, const float &height, const osg::Vec3 &pos,
                     const ColorMap &colorMap)
      : radius(rad), height(height), position(pos), colorMap(colorMap) {}
  CylinderAttributes(const float &rad, const float &height, const osg::Vec4 &maxCol,
                     const osg::Vec4 &minCol, const osg::Vec3 &pos,
                     const osg::Vec4 &defaultCol)
      : CylinderAttributes(rad, height, pos, ColorMap(maxCol, minCol, defaultCol)) {}
  CylinderAttributes(const float &rad, const float &height, const osg::Vec4 &maxCol,
                     const osg::Vec4 &minCol, const osg::Vec4 &defaultCol)
      : CylinderAttributes(rad, height, osg::Vec3(0, 0, 0),
                           ColorMap(maxCol, minCol, defaultCol)) {}
  float radius;
  float height;
  osg::Vec3 position;
  ColorMap colorMap;
};

/**
 * @class PrototypeBuilding
 * @brief Building prototype using cylinder attributes.
 *
 * Inherits IBuilding. Manages drawables and colors based on simulation state.
 */
class PrototypeBuilding : public core::interface::IBuilding {
 public:
  PrototypeBuilding(const CylinderAttributes &cylinderAttributes)
      : m_attributes(cylinderAttributes) {};
  void initDrawables() override;
  void updateColor(const osg::Vec4 &color) override;
  void updateTime(int timestep) override;
  void updateDrawables() override;
  std::unique_ptr<osg::Vec4> getColorInRange(float value, float maxValue) override;

 private:
  CylinderAttributes m_attributes;
};
