#include "TxtInfoboard.h"

#include <cover/coVRFileManager.h>
#include <lib/core/utils/osgUtils.h>

#include <osg/Billboard>
#include <osg/BlendFunc>
#include <osg/Depth>
#include <osg/Geode>
#include <osg/Group>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/Vec3>
#include <osg/ref_ptr>

#include "cover/coBillboard.h"
#include "lib/core/utils/color.h"

using namespace core;

void TxtInfoboard::updateTime(int timestep) {
  // TODO: implement later when needed
}

void TxtInfoboard::updateDrawable() {}

void TxtInfoboard::initDrawable() {
  osg::ref_ptr<osg::MatrixTransform> trans = new osg::MatrixTransform;
  trans->setMatrix(osg::Matrix::translate(m_attributes.position));
  trans->addChild(m_BBoard);
  trans->setName("Billboard");
  m_drawable = trans;
}

void TxtInfoboard::initInfoboard() {
  m_BBoard = new opencover::coBillboard();
  m_BBoard->setNormal(osg::Vec3(0, -1, 0));
  m_BBoard->setAxis(osg::Vec3(0, 0, 1));
  m_BBoard->setMode(opencover::coBillboard::AXIAL_ROT);
}

void TxtInfoboard::updateInfo(const std::string &info) {
  m_info = info;
  m_BBoard->removeChild(m_TextGeode);
  utils::osgUtils::deleteChildrenRecursive(m_BBoard);
  osg::Vec3 pos = osg::Vec3(0, 0, 0);
  auto contentPos = pos;
  contentPos.z() -= m_attributes.height * m_attributes.titleHeightPercentage;

  auto textBoxTitle = utils::osgUtils::createTextBox(
      m_attributes.title, pos, m_attributes.charSize, m_attributes.fontFile.c_str(),
      m_attributes.maxWidth, m_attributes.margin);
  auto textBoxContent = utils::osgUtils::createTextBox(
      "", contentPos, m_attributes.charSize, m_attributes.fontFile.c_str(),
      m_attributes.maxWidth, m_attributes.margin);
  textBoxContent->setText(info, osgText::String::ENCODING_UTF8);

  osg::ref_ptr<osg::Geode> geoText = new osg::Geode();
  geoText->setName("TextBox");
  geoText->addDrawable(textBoxTitle);
  geoText->addDrawable(textBoxContent);

  osg::Vec4 backgroundColor(0.2f, 0.2f, 0.2f, 0.8f);
  osg::BoundingBox bb;
  bb.expandBy(textBoxTitle->getBound());
  bb.expandBy(textBoxContent->getBound());

  // Create a rectangle for the background.
  float width = bb.xMax() - bb.xMin() + 2 * 0.5f;
  float height = bb.zMax() - bb.zMin() + 2 * 0.5f;
  osg::Vec3 center = bb.center();
  center.y() += 0.5f;
  auto backgroundGeometry = utils::osgUtils::createBackgroundQuadGeometry(
      center, width, height, backgroundColor);

  osg::ref_ptr<osg::Geode> geo = new osg::Geode();
  geo->setName("Background");
  geo->addDrawable(backgroundGeometry);
  utils::color::overrideGeodeColor(geo, backgroundColor,
                                   osg::Material::FRONT_AND_BACK);
  utils::osgUtils::setTransparency(geo, backgroundColor.a());
  utils::osgUtils::enableLighting(geo, false);

  m_TextGeode = new osg::Group();
  m_TextGeode->setName("TextGroup");
  m_TextGeode->addChild(geoText);
  m_TextGeode->addChild(geo);

  if (m_enabled) showInfo();
}

void TxtInfoboard::showInfo() {
  m_BBoard->addChild(m_TextGeode);
  m_enabled = true;
}

void TxtInfoboard::move(const osg::Vec3 &pos) {
  m_attributes.position = pos;
  if (m_enabled) updateInfo(m_info);
}

void TxtInfoboard::hideInfo() {
  m_BBoard->removeChild(m_TextGeode);
  m_enabled = false;
}
