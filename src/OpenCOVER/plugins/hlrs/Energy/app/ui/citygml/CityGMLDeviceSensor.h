#pragma once
#include <PluginUtil/coSensor.h>
#include <PluginUtil/colors/coColorMap.h>
#include <lib/core/interfaces/IBuilding.h>
#include <lib/core/interfaces/IInfoboard.h>
#include <lib/core/utils/color.h>

#include <memory>
#include <osg/Group>

/**
 * @class CityGMLDeviceSensor
 * @brief Represents a sensor for CityGML devices, providing interaction and visualization capabilities.
 *
 * This class extends coPickSensor to enable picking and interaction with CityGML buildings in an OpenCOVER scene.
 * It manages the visualization, color mapping, and info board display for a building, and allows updating
 * of timestep, color, and textual information.
 *
 * @note Copy construction and assignment are disabled.
 *
 * @param group The OSG group node to which the sensor is attached.
 * @param infoBoard Unique pointer to an info board interface for displaying textual information.
 * @param drawableBuilding Unique pointer to a building interface for drawable management.
 * @param textBoxTxt Optional vector of strings for initial text box content.
 *
 * @method update Updates the sensor state.
 * @method setTimestep Sets the current timestep for the building visualization.
 * @method activate Activates the sensor, enabling interaction.
 * @method disactivate Deactivates the sensor, disabling interaction.
 * @method updateTime Updates the building visualization to a specific timestep.
 * @method updateColorOfBuilding Updates the color of the building.
 * @method getDrawables Returns the drawables associated with the building.
 * @method getDrawable Returns a specific drawable node by index.
 * @method getParent Returns the parent group node.
 * @method updateTimestepColors Updates the colors of the building based on timestep values and a color map.
 * @method updateTxtBoxTexts Updates the text box content.
 * @method updateTitleOfInfoboard Updates the title of the info board.
 * @method setColorMapInShader Sets the color map in the building's shader.
 * @method setDataInShader Sets data values in the building's shader, with specified min and max.
 *
 * @private
 * @var m_cityGMLBuilding Unique pointer to the building interface.
 * @var m_infoBoard Unique pointer to the info board interface.
 * @var m_colors Vector of colors for visualization.
 * @var m_textBoxTxt Vector of strings for text box content.
 * @var m_active Indicates whether the sensor is active.
 */
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
