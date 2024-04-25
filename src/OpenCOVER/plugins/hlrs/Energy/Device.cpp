/* This file is part of COVISE.

  You can use it under the terms of the GNU Lesser General Public License
  version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "Device.h"
#include "DeviceSensor.h"
#include <cover/coVRFileManager.h>
#include <cstdio>
#include <osg/Material>

Device::Device(DeviceInfo *d, osg::Group *parent)
{
    myParent = parent;
    devInfo = d;

    deviceGroup = new osg::Group();
    deviceGroup->setName(devInfo->ID + ".");
    devSensor = new DeviceSensor(this, deviceGroup);
    myParent->addChild(deviceGroup);

    osg::MatrixTransform *matTrans = new osg::MatrixTransform();
    osg::Matrix m;
    m.makeTranslate(osg::Vec3(devInfo->lat, devInfo->lon, devInfo->height + h));
    matTrans->setMatrix(m);

    BBoard = new coBillboard();
    BBoard->setNormal(osg::Vec3(0, -1, 0));
    BBoard->setAxis(osg::Vec3(0, 0, 1));
    BBoard->setMode(coBillboard::AXIAL_ROT);

    matTrans->addChild(BBoard);
    deviceGroup->addChild(matTrans);
    TextGeode = nullptr;
}

Device::~Device()
{}

osg::Vec4 Device::getColor(float val, float max)
{
    osg::Vec4 colHigh = osg::Vec4(1, 0.1, 0, 1.0);
    osg::Vec4 colLow = osg::Vec4(0, 1, 0.5, 1.0);
    float valN = val / max;
    osg::Vec4 col(colHigh.r() * valN + colLow.r() * (1 - valN), colHigh.g() * valN + colLow.g() * (1 - valN),
                  colHigh.b() * valN + colLow.b() * (1 - valN), colHigh.a() * valN + colLow.a() * (1 - valN));
    return col;
}

void Device::init(float r, float sH, int c)
{
    if (geoBars) {
        deviceGroup->removeChild(geoBars);
        geoBars = nullptr;
    }

    rad = r;
    w = rad * 10;
    h = rad * 11;

    osg::Cylinder *cyl = new osg::Cylinder(osg::Vec3(devInfo->lat, devInfo->lon, devInfo->height), rad, 0.);
    osg::Vec4 colVec(0.1, 0.1, 0.1, 1.f);
    osg::Cylinder *cylLimit;
    osg::Vec4 colVecLimit(1.f, 1.f, 1.f, 1.f);

    auto setCyclAndColor = [&](const float &compVal) {
        cyl->set(osg::Vec3(devInfo->lat, devInfo->lon, devInfo->height + compVal * sH / 2), rad, -compVal * sH);
        colVec = getColor(compVal, 1000.);
    };

    switch (c) {
    case 0:
        if (devInfo->strom > 0.)
            setCyclAndColor(devInfo->strom);
        break;
    case 1:
        if (devInfo->waerme > 0.)
            setCyclAndColor(devInfo->waerme);
        break;
    case 2:
        if (devInfo->kaelte > 0.)
            setCyclAndColor(devInfo->kaelte);
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

    geoBars = new osg::Geode();
    geoBars->setName(devInfo->ID);
    geoBars->addDrawable(shapeD);

    deviceGroup->addChild(geoBars.get());
}

void Device::update()
{
    if (devSensor) {
        InfoVisible = false;
        devSensor->update();
    }
}
void Device::activate()
{
    if (TextGeode) {
        BBoard->removeChild(TextGeode);
        TextGeode = nullptr;
        InfoVisible = false;
    } else {
        showInfo();
        InfoVisible = true;
    }
}

void Device::disactivate()
{}

void Device::showInfo()
{
    osg::MatrixTransform *matShift = new osg::MatrixTransform();
    osg::Matrix ms;
    int charSize = 2;
    ms.makeTranslate(osg::Vec3(w / 2, 0, h));
    matShift->setMatrix(ms);
    osgText::Text *textBoxTitle = new osgText::Text();
    textBoxTitle->setAlignment(osgText::Text::LEFT_TOP);
    textBoxTitle->setAxisAlignment(osgText::Text::XZ_PLANE);
    textBoxTitle->setColor(osg::Vec4(1, 1, 1, 1));
    textBoxTitle->setText(devInfo->name, osgText::String::ENCODING_UTF8);
    textBoxTitle->setCharacterSize(charSize);
    textBoxTitle->setFont(coVRFileManager::instance()->getFontFile("DroidSans-Bold.ttf"));
    textBoxTitle->setMaximumWidth(w);
    textBoxTitle->setPosition(osg::Vec3(rad - w / 2., 0, h * 0.9));

    osgText::Text *textBoxContent = new osgText::Text();
    textBoxContent->setAlignment(osgText::Text::LEFT_TOP);
    textBoxContent->setAxisAlignment(osgText::Text::XZ_PLANE);
    textBoxContent->setColor(osg::Vec4(1.f, 1.f, 1.f, 1.f));
    textBoxContent->setLineSpacing(1.25);
    textBoxContent->setText(" > Baujahr:\n > Grundfläche:\n > Strom:\n > Wärme:\n > Kälte:",
                            osgText::String::ENCODING_UTF8);
    textBoxContent->setCharacterSize(charSize);
    textBoxContent->setFont(coVRFileManager::instance()->getFontFile(NULL));
    textBoxContent->setMaximumWidth(w * 2. / 3.);
    textBoxContent->setPosition(osg::Vec3(rad - w / 2.f, 0, h * 0.75));

    osgText::Text *textBoxValues = new osgText::Text();
    textBoxValues->setAlignment(osgText::Text::LEFT_TOP);
    textBoxValues->setAxisAlignment(osgText::Text::XZ_PLANE);
    textBoxValues->setColor(osg::Vec4(1.f, 1.f, 1.f, 1.f));
    textBoxValues->setLineSpacing(1.25);

    std::string textvalues = (devInfo->baujahr > 0.f ? (std::to_string((int)devInfo->baujahr) + " \n") : "- \n");
    textvalues += (devInfo->flaeche > 0.f ? (std::to_string((int)devInfo->flaeche) + " m2 \n") : "- \n");
    textvalues += (devInfo->strom < 0.f ? "- \n" : (std::to_string((int)devInfo->strom) + " MW\n"));
    textvalues += (devInfo->waerme < 0.f ? "- \n" : (std::to_string((int)devInfo->waerme) + " kW\n"));
    textvalues += (devInfo->kaelte < 0.f ? "- \n" : (std::to_string((int)devInfo->kaelte) + " kW\n"));

    textBoxValues->setText(textvalues);
    textBoxValues->setCharacterSize(charSize);
    textBoxValues->setFont(coVRFileManager::instance()->getFontFile(NULL));
    textBoxValues->setMaximumWidth(w / 3.);
    textBoxValues->setPosition(osg::Vec3(rad + w / 6., 0, h * 0.75));

    osg::Vec4 colVec(0., 0., 0., 0.2);
    osg::ref_ptr<osg::Material> mat = new osg::Material();
    mat->setDiffuse(osg::Material::FRONT_AND_BACK, colVec);
    mat->setAmbient(osg::Material::FRONT_AND_BACK, colVec);

    osg::Box *box = new osg::Box(osg::Vec3(rad, 0.04 * rad, h / 2.f), w, 0, h);
    osg::ShapeDrawable *sdBox = new osg::ShapeDrawable(box);
    sdBox->setColor(colVec);
    osg::StateSet *boxState = sdBox->getOrCreateStateSet();
    boxState->setAttribute(mat.get(), osg::StateAttribute::PROTECTED);
    sdBox->setStateSet(boxState);

    osg::StateSet *textStateT = textBoxTitle->getOrCreateStateSet();
    textBoxTitle->setStateSet(textStateT);
    osg::StateSet *textStateC = textBoxContent->getOrCreateStateSet();
    textBoxContent->setStateSet(textStateC);
    osg::StateSet *textStateV = textBoxValues->getOrCreateStateSet();
    textBoxValues->setStateSet(textStateV);

    osg::Geode *geo = new osg::Geode();
    geo->setName("TextBox");
    geo->addDrawable(textBoxTitle);
    geo->addDrawable(textBoxContent);
    geo->addDrawable(textBoxValues);
    geo->addDrawable(sdBox);

    matShift->addChild(geo);
    TextGeode = new osg::Group();
    TextGeode->setName("TextGroup");
    TextGeode->addChild(matShift);
    BBoard->addChild(TextGeode);
}
