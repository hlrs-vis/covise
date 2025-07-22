#include "CityGMLBuilding.h"

#include <lib/core/utils/color.h>

#include <cassert>
#include <iostream>
#include <osg/Geode>
#include <osg/MatrixTransform>

using namespace core::utils;

namespace {
constexpr auto SHADER_SCALAR_TIMESTEP_MAPPING_INDEX =
    0;  // index of the texture that maps from node index to timestep value
}

CityGMLBuilding::CityGMLBuilding(const osgUtils::Geodes &geodes) : m_shaders() {
  m_drawables.reserve(geodes.size());
  m_drawables.insert(m_drawables.begin(), geodes.begin(), geodes.end());
}

void CityGMLBuilding::initDrawables() {}

void CityGMLBuilding::updateColor(const osg::Vec4 &color) {
  for (auto drawable : m_drawables) {
    if (auto geo = drawable->asGeode()) color::overrideGeodeColor(geo, color);
  }
}

void CityGMLBuilding::updateTime(int timestep) {
  m_timestep = timestep;
  if (m_shaders.empty()) {
    std::cerr << "CityGMLBuilding::updateColor: "
              << "Shaders are not supported for CityGMLBuilding.\n";
    return;
  }

  for (auto i = 0; i < m_shaders.size(); ++i) {
    auto shader = m_shaders[i];
    auto geo = m_drawables[i]->asGeode();
    if (!shader || !geo) {
      std::cerr << "CityGMLBuilding::updateTime: "
                << "No shader or geode found for drawable at index " << i << "\n";
      continue;
    }
    shader->setIntUniform("timestep", timestep);
    auto state = geo->getOrCreateStateSet();
    shader->apply(state);
    geo->setStateSet(state);
  }
}

void CityGMLBuilding::updateDrawables() {}
std::unique_ptr<osg::Vec4> CityGMLBuilding::getColorInRange(float value,
                                                            float maxValue) {
  return nullptr;
}

void CityGMLBuilding::setColorMapInShader(const opencover::ColorMap &colorMap) {
  m_shaders.resize(m_drawables.size());
  for (auto i = 0; i < m_drawables.size(); ++i) {
    auto node = m_drawables[i];
    osg::ref_ptr<osg::Geode> geo = node->asGeode();
    if (!geo) {
      std::cerr << "CityGMLBuilding::setColorMapInShader: "
                << "No geode found for drawable.\n";
      continue;
    }
    auto shader = opencover::applyShader(geo, colorMap, "EnergyGrid");
    if (shader) {
      auto state = geo->getOrCreateStateSet();
      shader->apply(state);
      geo->setStateSet(state);
    }
    m_shaders[i] = shader;
  }
}

void CityGMLBuilding::setDataInShader(const std::vector<double> &data, float min,
                                      float max) {
  if (m_shaders.empty()) {
    std::cerr << "CityGMLBuilding::setData: No shader set for connection "
              << "\n";
    return;
  }

  for (auto i = 0; i < m_shaders.size(); ++i) {
    auto shader = m_shaders[i];
    // osg::ref_ptr<osg::Geode> geo = dynamic_cast<osg::Geode *>(m_drawables[i].get());
    osg::ref_ptr<osg::Geode> geo = m_drawables[i]->asGeode();
    if (!shader || !geo) {
      std::cerr << "CityGMLBuilding::setData: "
                << "No shader or geode found for drawable at index " << i << "\n";
      continue;
    }
    shader->setIntUniform("numTimesteps", data.size());
    shader->setIntUniform("numNodes", 1);
    auto uniform = shader->getcoVRUniform("timestepToData");
    assert(uniform);
    uniform->setValue(std::to_string(SHADER_SCALAR_TIMESTEP_MAPPING_INDEX).c_str());

    auto texture = core::utils::osgUtils::createValue1DTexture(data);
    auto state = geo->getOrCreateStateSet();
    state->setTextureAttribute(SHADER_SCALAR_TIMESTEP_MAPPING_INDEX, texture,
                               osg::StateAttribute::ON);

    shader->apply(state);
    geo->setStateSet(state);
  }
}
