/* This file is part of COVISE.

  You can use it under the terms of the GNU Lesser General Public License
  version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

/****************************************************************************\
 **                                                          (C)2020 HLRS  **
 **                                                                        **
 ** Description: OpenCOVER Plug-In for reading Luftdaten sensor data       **
 **                                                                        **
 **                                                                        **
 ** Author: Leyla Kern                                                     **
 **                                                                        **
 ** History:                                                               **
 ** April 2020  v1                                                         **
 **                                                                        **
 **                                                                        **
\****************************************************************************/
#include "Device.h"
#include "DeviceSensor.h"
#include <osg/Material>
#include <cover/coVRFileManager.h>

Device::Device(DeviceInfo *d, osg::Group *parent)
{
    myParent = parent;
    devInfo = d;

    deviceGroup = new osg::Group();
    deviceGroup->setName(devInfo->ID+".");
    devSensor = new DeviceSensor(this,deviceGroup);
    myParent->addChild(deviceGroup);
    
    osg::MatrixTransform* matTrans = new osg::MatrixTransform();
    osg::Matrix m;
    m.makeTranslate(osg::Vec3(devInfo->lat, devInfo->lon, devInfo->height+h));
    matTrans->setMatrix(m);
    
    BBoard = new coBillboard();
    BBoard->setNormal(osg::Vec3(0,-1,0));
    BBoard->setAxis(osg::Vec3(0,0,1));
    BBoard->setMode(coBillboard::AXIAL_ROT);
    
    matTrans->addChild(BBoard);
    deviceGroup->addChild(matTrans);
    TextGeode = nullptr;
}
Device::~Device()
{
    
}
void Device::init(float r, float sH,int c)
{
    if (geoBars)
    {
        deviceGroup->removeChild(geoBars);
        geoBars = nullptr;
    }
    
    rad = r;
    w = rad * 10;
    h = rad * 11;
    osg::Cylinder * cyl = new osg::Cylinder(osg::Vec3(devInfo->lat, devInfo->lon,devInfo->height),rad,0.);
    osg::Vec4 colVec(0.1,0.1,0.1,1.f);
    osg::Cylinder * cylLimit;
    osg::Vec4 colVecLimit(1.f,1.f,1.f,1.f);
    
    switch (c) {
    case 0:
        if (devInfo->pm10 > -1.)
        {
            cyl->set(osg::Vec3(devInfo->lat, devInfo->lon,devInfo->height+devInfo->pm10*sH/2),rad,-devInfo->pm10*sH);
            colVec = osg::Vec4(devInfo->pm10/70.f,1-devInfo->pm10/70.f,0.f,1.f);
            if (devInfo->pm10 > 50)
                cylLimit = new osg::Cylinder(osg::Vec3(devInfo->lat, devInfo->lon,devInfo->height + 50*sH),rad*1.01,-1*sH);
        }
        break;
    case 1:
        if (devInfo->pm2  > -1.)
        {
            cyl->set(osg::Vec3(devInfo->lat, devInfo->lon,devInfo->height+devInfo->pm2*sH/2),rad,-devInfo->pm2*sH);
            colVec = osg::Vec4(devInfo->pm2/20.f,1-devInfo->pm2/20.f,0.f,1.f);
        }
        break;
    case 2:
        if (devInfo->temp > -100.)
        {
            cyl->set(osg::Vec3(devInfo->lat, devInfo->lon,devInfo->height+devInfo->temp*sH/2),rad,-devInfo->temp*sH);
            colVec = osg::Vec4(devInfo->temp/30.f,1-devInfo->temp/30.f,0.f,1.f);
        }
        break;
    case 3:
        if (devInfo->humi > -1.)
        {
            cyl->set(osg::Vec3(devInfo->lat, devInfo->lon,devInfo->height+devInfo->humi*sH/2),rad,-devInfo->humi*sH);
            colVec = osg::Vec4(devInfo->humi/70.f,1-devInfo->humi/70.f,0.f,1.f);
            if (devInfo->humi > 60)
                cylLimit = new osg::Cylinder(osg::Vec3(devInfo->lat, devInfo->lon,devInfo->height + 60*sH),rad*1.01,-1*sH);
        }
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
    state->setAttribute(mat.get(),osg::StateAttribute::PROTECTED);
    state->setNestRenderBins(false);
    
    shapeD->setStateSet(state);
    shapeD->setUseDisplayList(false);
    shapeD->setColor(colVec);

    geoBars = new osg::Geode();
    geoBars->setName(devInfo->ID);
    geoBars->addDrawable(shapeD);

    if ((devInfo->pm10 > 50 && c==0) || (devInfo->humi > 60 && c==3))
    {
        osg::ShapeDrawable *shapeDLimit = new osg::ShapeDrawable(cylLimit);
        osg::StateSet *stateLimit = shapeDLimit->getOrCreateStateSet();
        stateLimit->setAttribute(mat.get(),osg::StateAttribute::PROTECTED);
        stateLimit->setNestRenderBins(false);
        shapeDLimit->setStateSet(stateLimit);
        shapeDLimit->setUseDisplayList(false);
        shapeDLimit->setColor(colVecLimit);
        geoBars->addDrawable(shapeDLimit);
    }
    
    deviceGroup->addChild(geoBars.get());
}
void Device::showGraph()
{
    
}

void Device::update()
{
    if(devSensor)
    {
        InfoVisible = false;
        devSensor->update();
    }
}
void Device::activate()
{
    if (TextGeode)
    {
        BBoard->removeChild(TextGeode);
        TextGeode = nullptr;
        InfoVisible = false;
    }else {
        showInfo();
        InfoVisible = true;
    }
}
void Device::disactivate()
{
    
}
void Device::showInfo()
{
    osg::MatrixTransform* matShift = new osg::MatrixTransform();
    osg::Matrix ms;
    ms.makeTranslate(osg::Vec3(w/2,0, h));
    matShift->setMatrix(ms);
    osgText::Text *textBoxTitle = new osgText::Text();
    textBoxTitle->setAlignment(osgText::Text::LEFT_TOP);
    textBoxTitle->setAxisAlignment(osgText::Text::XZ_PLANE );
    textBoxTitle->setColor(osg::Vec4(1, 1, 1, 1));
    textBoxTitle->setText("Sensor "+devInfo->ID);
    textBoxTitle->setCharacterSize(12);
    textBoxTitle->setFont(coVRFileManager::instance()->getFontFile("DroidSans-Bold.ttf"));
    textBoxTitle->setMaximumWidth(w);
    textBoxTitle->setPosition(osg::Vec3(rad-w/2.,0,h*0.9));

    osgText::Text *textBoxContent = new osgText::Text();
    textBoxContent->setAlignment(osgText::Text::LEFT_TOP );
    textBoxContent->setAxisAlignment(osgText::Text::XZ_PLANE );
    textBoxContent->setColor(osg::Vec4(1.f, 1.f, 1.f, 1.f));
    textBoxContent->setLineSpacing(1.25);
    textBoxContent->setText(" > PM10:\n > PM2.5:\n > Humidity:\n > Temperature:");
    textBoxContent->setCharacterSize(12);
    textBoxContent->setFont(coVRFileManager::instance()->getFontFile(NULL));
    textBoxContent->setMaximumWidth(w*2./3.);
    textBoxContent->setPosition(osg::Vec3(rad-w/2.f,0,h*0.75));
    
    osgText::Text *textBoxValues = new osgText::Text();
    textBoxValues->setAlignment(osgText::Text::LEFT_TOP );
    textBoxValues->setAxisAlignment(osgText::Text::XZ_PLANE );
    textBoxValues->setColor(osg::Vec4(1.f, 1.f, 1.f, 1.f));
    textBoxValues->setLineSpacing(1.25);
    std::string textvalues = (devInfo->pm10 < 0.f ? "- \n" : (std::to_string((int)devInfo->pm10) + " \n"));
    textvalues += (devInfo->pm2 < 0.f ? "- \n" : (std::to_string((int)devInfo->pm2) + " \n"));
    textvalues += (devInfo->humi < 0.f ? "- \n" : (std::to_string((int)devInfo->humi) + " \n"));
    textvalues += (devInfo->temp < -99.f ? "- \n" : (std::to_string((int)devInfo->temp) + " \n"));
    textBoxValues->setText(textvalues);
    textBoxValues->setCharacterSize(12);
    textBoxValues->setFont(coVRFileManager::instance()->getFontFile(NULL));
    textBoxValues->setMaximumWidth(w/3.);
    textBoxValues->setPosition(osg::Vec3(rad+w/6.,0,h*0.75));
    
    osg::Vec4 colVec(0.,0.,0.,0.2);
    osg::ref_ptr<osg::Material> mat = new osg::Material();
    mat->setDiffuse(osg::Material::FRONT_AND_BACK, colVec);
    mat->setAmbient(osg::Material::FRONT_AND_BACK, colVec);
    
    osg::Box *box = new osg::Box(osg::Vec3(rad,0.04*rad,h/2.f),w,0,h);
    osg::ShapeDrawable *sdBox = new osg::ShapeDrawable(box);
    sdBox->setColor(colVec);
    osg::StateSet *boxState = sdBox->getOrCreateStateSet();
    boxState->setAttribute(mat.get(),osg::StateAttribute::PROTECTED);
    sdBox->setStateSet(boxState);

    osg::StateSet *textStateT = textBoxTitle->getOrCreateStateSet();
    textBoxTitle->setStateSet(textStateT);
    osg::StateSet *textStateC = textBoxContent->getOrCreateStateSet();
    textBoxContent->setStateSet(textStateC);
    osg::StateSet *textStateV = textBoxValues->getOrCreateStateSet();
    textBoxValues->setStateSet(textStateV);
    
    osg::Geode *geo= new osg::Geode();
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



