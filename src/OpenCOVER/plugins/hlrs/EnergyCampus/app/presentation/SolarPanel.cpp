#include "SolarPanel.h"

#include <cover/coVRShader.h>
#include <lib/core/utils/color.h>
#include <lib/core/utils/osgUtils.h>

#include <iostream>
#include <osg/Array>
#include <osg/Drawable>
#include <osg/Geode>
#include <osg/Material>
#include <osg/Program>
#include <osg/Shader>
#include <osg/State>
#include <osg/StateAttribute>
#include <osg/StateSet>
#include <osg/Texture2D>
#include <osg/Vec4>
#include <osg/ref_ptr>
#include <string>

using namespace core::utils::osgUtils;

void SolarPanel::init() { initDrawables(); }

void SolarPanel::initDrawables() {
  osg::ref_ptr<osg::Group> group = m_node->asGroup();
  if (!group) {
    std::cerr << "SolarPanel: m_node is not a group!" << std::endl;
    return;
  }
  auto geodes = getGeodes(group);
  std::string name = "SolarpanelPart";
  int i = 0;
  for (auto geode : *geodes) {
    geode->setName(name + std::to_string(i));
    ++i;
  }
  m_drawables.push_back(m_node);
}

void SolarPanel::updateDrawables() {}

void SolarPanel::updateColor(const osg::Vec4 &color) {
  for (auto &node : m_drawables) {
    auto geode = dynamic_cast<osg::Geode *>(node.get());
    if (geode) {
      core::utils::color::overrideGeodeColor(geode, color);
      continue;
    }

    auto group = dynamic_cast<osg::Group *>(node.get());
    if (group) {
      auto geodes = getGeodes(group);
      for (auto &geode : *geodes) {
        osg::ref_ptr<osg::Material> mat = new osg::Material;
        osg::Material::Face face = osg::Material::FRONT_AND_BACK;
        mat->setDiffuse(face, color);
        mat->setAmbient(face, color);
        mat->setEmission(face, color);
        osg::ref_ptr<osg::StateSet> stateset = geode->getOrCreateStateSet();
        if (!stateset) continue;
        stateset->setAttribute(mat, osg::StateAttribute::OVERRIDE);
        stateset->setMode(GL_LIGHTING, osg::StateAttribute::ON);
      }
    }
  }
}
