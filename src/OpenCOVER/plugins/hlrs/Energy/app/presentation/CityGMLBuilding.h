#pragma once
#include <lib/core/interfaces/IBuilding.h>
#include <lib/core/utils/osgUtils.h>
#include <PluginUtil/coShaderUtil.h>

#include <memory>

/**
 * @class CityGMLBuilding
 * @brief Represents a building entity based on CityGML data, providing rendering and data mapping functionalities.
 *
 * Inherits from core::interface::IBuilding and manages graphical representation using OpenSceneGraph (OSG) geodes.
 * Supports shader-based coloring and data visualization.
 *
 * @constructor CityGMLBuilding(const core::utils::osgUtils::Geodes &geodes)
 *   Constructs a CityGMLBuilding with the specified OSG geodes.
 *
 * @fn void initDrawables() override
 *   Initializes drawable objects for rendering the building.
 *
 * @fn void updateColor(const osg::Vec4 &color) override
 *   Updates the color of the building using the specified RGBA value.
 *
 * @fn void updateTime(int timestep) override
 *   Updates the building's state based on the given timestep.
 *
 * @fn void updateDrawables() override
 *   Refreshes the drawable objects, typically after data or state changes.
 *
 * @fn std::unique_ptr<osg::Vec4> getColorInRange(float value, float maxValue) override
 *   Computes and returns a color corresponding to the given value within a specified range.
 *
 * @fn void setColorMapInShader(const opencover::ColorMap &colorMap)
 *   Sets the color map in the building's shaders for data visualization.
 *
 * @fn void setDataInShader(const std::vector<double> &data, float min, float max)
 *   Passes data and its range to the shaders for visualization purposes.
 *
 * @fn bool hasShader() const
 *   Checks if the building has any associated shaders.
 *
 * @var std::vector<opencover::coVRShader *> m_shaders
 *   Stores pointers to shaders used for rendering and data mapping.
 */
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
