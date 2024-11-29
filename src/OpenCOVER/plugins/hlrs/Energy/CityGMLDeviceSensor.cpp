#include <CityGMLDeviceSensor.h>
#include <PluginUtil/coColorMap.h>
#include <PluginUtil/coSensor.h>
#include <PluginUtil/coShaderUtil.h>
#include <core/CityGMLBuilding.h>

#include <cstdint>
#include <memory>
#include <osg/Geometry>

namespace {
constexpr const char *SHADER_FILE = "MapColorsAttrib";
}

CityGMLDeviceSensor::CityGMLDeviceSensor(
    osg::ref_ptr<osg::Group> parent,
    std::unique_ptr<core::interface::IInfoboard<std::string>> &&infoBoard,
    std::unique_ptr<core::interface::IBuilding> &&drawableBuilding,
    std::shared_ptr<ColorMapExtended> colorMap)
    : coPickSensor(parent),
      m_cityGMLBuilding(std::move(drawableBuilding)),
      m_infoBoard(std::move(infoBoard)),
      m_colorMapRef(colorMap) {
  m_cityGMLBuilding->initDrawables();

  // infoboard
  m_infoBoard->initInfoboard();
  m_infoBoard->initDrawable();
  parent->addChild(m_infoBoard->getDrawable());
}

CityGMLDeviceSensor::~CityGMLDeviceSensor() {
  if (m_active) disactivate();
  getParent()->removeChild(m_infoBoard->getDrawable());
}

void CityGMLDeviceSensor::update() {
  m_cityGMLBuilding->updateDrawables();
  m_infoBoard->updateDrawable();
  coPickSensor::update();
}

void CityGMLDeviceSensor::activate() {
  if (!m_active) {
    m_infoBoard->updateInfo("DAS IST EIN TEST");
    m_infoBoard->showInfo();
  }
  m_active = !m_active;
}

void CityGMLDeviceSensor::disactivate() {
  if (m_active) return;
  m_infoBoard->hideInfo();
}

// void CityGMLDeviceSensor::updateShader() {
//   auto color_map = m_colorMapRef.lock();
//   for (auto node : m_cityGMLBuilding->getDrawables()) {
//     osg::ref_ptr<osg::Geode> geode = node->asGeode();
//     osg::ref_ptr<osg::Drawable> drawable = geode->getDrawable(0);
//     opencover::applyShader(drawable->asDrawable(), *color_map, 0, 100,
//     SHADER_FILE);
//   }
// }

void CityGMLDeviceSensor::updateTime(int timestep) {
  static std::uint8_t r = 255;
  static std::uint8_t g = 0;
  static std::uint8_t b = 0;

  if (r == 255 && g < 255 && b == 0) {
    g++;
  } else if (r > 0 && g == 255 && b == 0) {
    r--;
  } else if (r == 0 && g == 255 && b < 255) {
    b++;
  } else if (r == 0 && g > 0 && b == 255) {
    g--;
  } else if (r < 255 && g == 0 && b == 255) {
    r++;
  } else if (r == 255 && g == 0 && b > 0) {
    b--;
  }

  //   osg::Vec4 color = osg::Vec4(r / 255.0, g / 255.0, b / 255.0, 1.0);

  //   m_cityGMLBuilding->updateColor(osg::Vec4(r / 255.0, g / 255.0, b /
  //   255.0, 1.0));
  //   int val = rand() % 255 / 255;

  auto color_map = m_colorMapRef.lock();
  auto val = r / 255.0;
  if (val >= color_map->min && val <= color_map->max) {
    auto color = covise::getColor(val, color_map->map, color_map->min,
                                  color_map->max);
    //   opencover::applyShader(m_cityGMLBuilding->get, const covise::ColorMap
    //   &colorMap, float min, float max, const std::string &shaderFile)
    m_cityGMLBuilding->updateColor(color);
  }
  m_cityGMLBuilding->updateTime(timestep);
  m_infoBoard->updateTime(timestep);
}
