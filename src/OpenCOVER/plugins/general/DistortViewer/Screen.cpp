/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Screen.h"
#include "ScreenDome.h"
#include "ScreenCylinder.h"
#include "ScreenPlane.h"
#include "XmlTools.h"

// Konstruktor
Screen::Screen()
    : stateMesh(false)
    , orientation(osg::Vec3(0.0f, 0.0f, 0.0f))
    , centerPos(osg::Vec3(0.0f, 0.0f, 0.0f))
    , scaleVec(osg::Vec3(1.0f, 1.0f, 1.0f))

{
    calcTransMat();
}

// Destruktor
Screen::~Screen()
{
}

void Screen::saveToXML()
{
    std::string section;
    std::string var_str;
    std::string plugPath = XmlTools::getInstance()->getPlugPath();

    //Geometry
    section = "Geometry";

    XmlTools::getInstance()->saveStrValue(geoShape, plugPath + "." + section, "Shape");
    XmlTools::getInstance()->saveVec3(orientation, plugPath + "." + section + ".OrientationVec3");
    XmlTools::getInstance()->saveVec3(centerPos, plugPath + "." + section + ".PositionVec3");
    XmlTools::getInstance()->saveVec3(scaleVec, plugPath + "." + section + ".SizeVec3");
    XmlTools::getInstance()->saveBoolValue(stateMesh, plugPath + "." + section, "MeshChk");

    //Einträge in xml schreiben
    XmlTools::getInstance()->saveToXml();
}

void Screen::loadFromXML()
{
    std::string section;
    std::string var_str;
    std::string plugPath = XmlTools::getInstance()->getPlugPath();

    //Geometry
    section = "Geometry";
    geoShape = XmlTools::getInstance()->loadStrValue(plugPath + "." + section, "Shape", "Dome");
    orientation = XmlTools::getInstance()->loadVec3(plugPath + "." + section + ".OrientationVec3", osg::Vec3(0.0f, 0.0f, 0.0f));
    centerPos = XmlTools::getInstance()->loadVec3(plugPath + "." + section + ".PositionVec3", osg::Vec3(0.0f, 0.0f, 0.0f));
    scaleVec = XmlTools::getInstance()->loadVec3(plugPath + "." + section + ".SizeVec3", osg::Vec3(1.0f, 1.0f, 1.0f));
    stateMesh = XmlTools::getInstance()->loadBoolValue(plugPath + "." + section, "MeshChk", true);

    calcTransMat();
}

void Screen::setCenterPos(osg::Vec3 new_centerPos)
{
    centerPos = new_centerPos;
    calcTransMat();
}

void Screen::setOrientation(osg::Vec3 new_orientation)
{
    orientation = new_orientation;
    calcTransMat();
}

void Screen::setScaleVec(osg::Vec3 new_scaleVec)
{
    scaleVec = new_scaleVec;
    calcTransMat();
}

void Screen::calcTransMat()
{
    sTransMat = osg::Matrix::identity()
                * osg::Matrix::rotate(osg::DegreesToRadians(orientation.x()), osg::X_AXIS,
                                      osg::DegreesToRadians(orientation.y()), osg::Y_AXIS,
                                      osg::DegreesToRadians(orientation.z()), osg::Z_AXIS)
                * osg::Matrix::translate(centerPos)
                * osg::Matrix::scale(scaleVec);
}

void Screen::setStateMesh(bool new_state)
{
    stateMesh = new_state;
}

osg::MatrixTransform *Screen::draw(bool gitter)
{
    //Geometrie-Node erstellen
    osg::ref_ptr<osg::Geode> geodeScreen = drawScreen(gitter);

    //ProjGeometrie Transformieren
    osg::ref_ptr<osg::MatrixTransform> transScreen = new osg::MatrixTransform();
    transScreen->setMatrix(sTransMat);
    transScreen->addChild(geodeScreen.get());

    return transScreen.release();
}