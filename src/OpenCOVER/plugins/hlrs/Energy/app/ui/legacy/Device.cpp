/* This file is part of COVISE.

  You can use it under the terms of the GNU Lesser General Public License
  version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "Device.h"

#include <cover/coVRFileManager.h>
#include <lib/core/constants.h>

#include <cstdio>
#include <osg/Material>

using namespace opencover;

namespace {
osg::Vec4 getColor(float val, float max) {
  osg::Vec4 colHigh = osg::Vec4(1, 0.1, 0, 1.0);
  osg::Vec4 colLow = osg::Vec4(0, 1, 0.5, 1.0);
  float valN = val / max;
  osg::Vec4 col(colHigh.r() * valN + colLow.r() * (1 - valN),
                colHigh.g() * valN + colLow.g() * (1 - valN),
                colHigh.b() * valN + colLow.b() * (1 - valN),
                colHigh.a() * valN + colLow.a() * (1 - valN));
  return col;
}
}  // namespace

namespace energy {
Device::Device(osg::ref_ptr<osg::MatrixTransform> node, const DeviceInfo &deviceInfo,
               const std::string &font)
    : m_devInfo(deviceInfo),
      m_font(font),
      m_height(1.0f),
      m_width(2.0f),
      m_node(new osg::MatrixTransform()),
      m_BBoard(new opencover::coBillboard()),
      m_infoVisible(false) {
  m_BBoard->setNormal(osg::Vec3(0, -1, 0));
  m_BBoard->setAxis(osg::Vec3(0, 0, 1));
  m_BBoard->setMode(opencover::coBillboard::AXIAL_ROT);

  m_node->addChild(m_BBoard);
  m_txtGroup = nullptr;
}

Device::~Device() {}

void Device::init(float r, float sH, int c) {
  if (m_geoBars) {
    m_node->removeChild(m_geoBars);
    m_geoBars = nullptr;
  }

  m_rad = r;
  m_width = m_rad * 10;
  m_height = m_rad * 11;

  osg::Cylinder *cyl = new osg::Cylinder(
      osg::Vec3(m_devInfo.lon, m_devInfo.lat, m_devInfo.height), m_rad, 0.);
  osg::Vec4 colVec(0.1, 0.1, 0.1, 1.f);
  osg::Cylinder *cylLimit;
  osg::Vec4 colVecLimit(1.f, 1.f, 1.f, 1.f);

  auto setCyclAndColor = [&](const float &compVal) {
    cyl->set(
        osg::Vec3(m_devInfo.lon, m_devInfo.lat, m_devInfo.height + compVal * sH / 2),
        m_rad, -compVal * sH);
    colVec = getColor(compVal, 1000.);
  };

  switch (c) {
    case 0:
      if (m_devInfo.strom > 0.) setCyclAndColor(m_devInfo.strom);
      break;
    case 1:
      if (m_devInfo.waerme > 0.) setCyclAndColor(m_devInfo.waerme);
      break;
    case 2:
      if (m_devInfo.kaelte > 0.) setCyclAndColor(m_devInfo.kaelte);
      break;
  }
  osg::ShapeDrawable *shapeD = new osg::ShapeDrawable(cyl);

  osg::ref_ptr<osg::Material> mat = new osg::Material();
  mat->setDiffuse(osg::Material::FRONT_AND_BACK, colVec);
  mat->setAmbient(osg::Material::FRONT_AND_BACK, colVec);
  mat->setEmission(osg::Material::FRONT_AND_BACK, colVec);
  mat->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
  mat->setColorMode(osg::Material::EMISSION);

  osg::StateSet *state = shapeD->getOrCreateStateSet();
  state->setAttribute(mat.get(), osg::StateAttribute::PROTECTED);
  state->setNestRenderBins(false);

  shapeD->setStateSet(state);
  shapeD->setUseDisplayList(false);
  shapeD->setColor(colVec);

  m_geoBars = new osg::Geode();
  m_geoBars->setName(m_devInfo.ID);
  m_geoBars->addDrawable(shapeD);

  m_node->addChild(m_geoBars.get());
}

void Device::update() { m_infoVisible = false; }

void Device::activate() {
  if (m_txtGroup) {
    m_BBoard->removeChild(m_txtGroup);
    m_txtGroup = nullptr;
    m_infoVisible = false;
  } else {
    showInfo();
    m_infoVisible = true;
  }
}

void Device::disactivate() {}

void Device::showInfo() {
  osg::ref_ptr<osg::MatrixTransform> matShift = new osg::MatrixTransform();
  osg::Matrix ms;
  int charSize = 2;
  ms.makeTranslate(osg::Vec3(m_width / 2, 0, m_height));
  matShift->setMatrix(ms);
  osg::ref_ptr<osgText::Text> textBoxTitle = new osgText::Text();
  textBoxTitle->setAlignment(osgText::Text::LEFT_TOP);
  textBoxTitle->setAxisAlignment(osgText::Text::XZ_PLANE);
  textBoxTitle->setColor(osg::Vec4(1, 1, 1, 1));
  textBoxTitle->setText(m_devInfo.name, osgText::String::ENCODING_UTF8);
  textBoxTitle->setCharacterSize(charSize);
  textBoxTitle->setFont(coVRFileManager::instance()->getFontFile(m_font.c_str()));
  textBoxTitle->setMaximumWidth(m_width);
  textBoxTitle->setPosition(osg::Vec3(m_rad - m_width / 2., 0, m_height * 0.9));

  osg::ref_ptr<osgText::Text> textBoxContent = new osgText::Text();
  textBoxContent->setAlignment(osgText::Text::LEFT_TOP);
  textBoxContent->setAxisAlignment(osgText::Text::XZ_PLANE);
  textBoxContent->setColor(osg::Vec4(1.f, 1.f, 1.f, 1.f));
  textBoxContent->setLineSpacing(1.25);
  std::array<std::string, 5> labels = {"Baujahr", "Grundfläche", "Strom", "Wärme",
                                       "Kälte"};
  std::string description("");
  for (const auto &label : labels)
    description += UIConstants::TAB_SPACES + label + ": \n";
  textBoxContent->setText(description, osgText::String::ENCODING_UTF8);
  textBoxContent->setCharacterSize(charSize);
  textBoxContent->setFont(coVRFileManager::instance()->getFontFile(NULL));
  textBoxContent->setMaximumWidth(m_width * 2. / 3.);
  textBoxContent->setPosition(osg::Vec3(m_rad - m_width / 2.f, 0, m_height * 0.75));

  osg::ref_ptr<osgText::Text> textBoxValues = new osgText::Text();
  textBoxValues->setAlignment(osgText::Text::LEFT_TOP);
  textBoxValues->setAxisAlignment(osgText::Text::XZ_PLANE);
  textBoxValues->setColor(osg::Vec4(1.f, 1.f, 1.f, 1.f));
  textBoxValues->setLineSpacing(1.25);

  std::string textvalues =
      (m_devInfo.baujahr > 0.f ? (std::to_string((int)m_devInfo.baujahr) + " \n")
                               : "- \n");
  textvalues +=
      (m_devInfo.flaeche > 0.f ? (std::to_string((int)m_devInfo.flaeche) + " m2 \n")
                               : "- \n");
  textvalues +=
      (m_devInfo.strom < 0.f ? "- \n"
                             : (std::to_string((int)m_devInfo.strom) + " MW\n"));
  textvalues +=
      (m_devInfo.waerme < 0.f ? "- \n"
                              : (std::to_string((int)m_devInfo.waerme) + " kW\n"));
  textvalues +=
      (m_devInfo.kaelte < 0.f ? "- \n"
                              : (std::to_string((int)m_devInfo.kaelte) + " kW\n"));

  textBoxValues->setText(textvalues);
  textBoxValues->setCharacterSize(charSize);
  textBoxValues->setFont(coVRFileManager::instance()->getFontFile(NULL));
  textBoxValues->setMaximumWidth(m_width / 3.);
  textBoxValues->setPosition(osg::Vec3(m_rad + m_width / 6., 0, m_height * 0.75));

  osg::Vec4 colVec(0., 0., 0., 0.2);
  osg::ref_ptr<osg::Material> mat = new osg::Material();
  mat->setDiffuse(osg::Material::FRONT_AND_BACK, colVec);
  mat->setAmbient(osg::Material::FRONT_AND_BACK, colVec);

  osg::ref_ptr<osg::Box> box = new osg::Box(
      osg::Vec3(m_rad, 0.04 * m_rad, m_height / 2.f), m_width, 0, m_height);
  osg::ShapeDrawable *sdBox = new osg::ShapeDrawable(box);
  sdBox->setColor(colVec);
  osg::ref_ptr<osg::StateSet> boxState = sdBox->getOrCreateStateSet();
  boxState->setAttribute(mat.get(), osg::StateAttribute::PROTECTED);
  sdBox->setStateSet(boxState);

  osg::ref_ptr<osg::StateSet> textStateT = textBoxTitle->getOrCreateStateSet();
  textBoxTitle->setStateSet(textStateT);
  osg::ref_ptr<osg::StateSet> textStateC = textBoxContent->getOrCreateStateSet();
  textBoxContent->setStateSet(textStateC);
  osg::ref_ptr<osg::StateSet> textStateV = textBoxValues->getOrCreateStateSet();
  textBoxValues->setStateSet(textStateV);

  osg::ref_ptr<osg::Geode> geo = new osg::Geode();
  geo->setName("TextBox");
  geo->addDrawable(textBoxTitle);
  geo->addDrawable(textBoxContent);
  geo->addDrawable(textBoxValues);
  geo->addDrawable(sdBox);

  matShift->addChild(geo);
  m_txtGroup = new osg::Group();
  m_txtGroup->setName("TextGroup");
  m_txtGroup->addChild(matShift);
  m_BBoard->addChild(m_txtGroup);
}
}  // namespace energy
