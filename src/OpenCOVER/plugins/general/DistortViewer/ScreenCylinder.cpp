/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ScreenCylinder.h"
#include "XmlTools.h"

ScreenCylinder::ScreenCylinder()
    : cRadius(2500.0f)
    , cHeight(5000.0f)
    , cSegSize(180.0f)
    , cZResolution(100)
    , cAzimResolution(100)
{
    geoShape = "Cylinder";
}

ScreenCylinder::~ScreenCylinder(void)
{
}

void ScreenCylinder::saveToXML()
{
    Screen::saveToXML();

    std::string var_str;
    std::string section = "Geometry";
    std::string subsection = geoShape;
    std::string plugPath = XmlTools::getInstance()->getPlugPath();
    std::string path = plugPath + "." + section + "." + subsection;

    XmlTools::getInstance()->saveFloatValue(cRadius, path, "Radius");
    XmlTools::getInstance()->saveFloatValue(cHeight, path, "Height");
    XmlTools::getInstance()->saveIntValue(cZResolution, path, "HeightRes");
    XmlTools::getInstance()->saveFloatValue(cSegSize, path, "AzimAngle");
    XmlTools::getInstance()->saveIntValue(cAzimResolution, path, "AzimRes");

    //Einträge in xml schreiben
    XmlTools::getInstance()->saveToXml();
}

void ScreenCylinder::loadFromXML()
{
    Screen::loadFromXML();

    std::string section = "Geometry";
    std::string subsection = geoShape;
    std::string plugPath = XmlTools::getInstance()->getPlugPath();
    std::string path = plugPath + "." + section + "." + subsection;

    cRadius = XmlTools::getInstance()->loadFloatValue(path, "Radius", 2500.0f);
    cHeight = XmlTools::getInstance()->loadFloatValue(path, "Height", 5000.0f);
    cZResolution = XmlTools::getInstance()->loadIntValue(path, "HeightRes", 100);
    cSegSize = XmlTools::getInstance()->loadFloatValue(path, "AzimAngle", 180.0f);
    cAzimResolution = XmlTools::getInstance()->loadIntValue(path, "AzimRes", 100);
}

void ScreenCylinder::setRadius(float radius)
{
    cRadius = radius;
}

void ScreenCylinder::setHeight(float height)
{
    cHeight = height;
}

void ScreenCylinder::setSegmentSize(float segSize)
{
    cSegSize = segSize;
}

void ScreenCylinder::setZResolution(unsigned int resolution)
{
    cZResolution = resolution;
}

void ScreenCylinder::setAzimResolution(unsigned int resolution)
{
    cAzimResolution = resolution;
}

osg::Geode *ScreenCylinder::drawScreen(bool gitter)
{
    float deltaAzim = (2 * osg::PI * (cSegSize / 360)) / (cAzimResolution);
    float deltaHeight = cHeight / (cZResolution);
    osg::ref_ptr<osg::Geometry> cylinderGeometry = new osg::Geometry();
    osg::ref_ptr<osg::Vec3Array> cylinderVertices = new osg::Vec3Array;
    osg::ref_ptr<osg::Vec3Array> cylinderNormals = new osg::Vec3Array;
    osg::ref_ptr<osg::Vec4Array> cylinderColor = new osg::Vec4Array;
    cylinderColor->push_back(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));

    //Vertices über Polarkoordinaten festlegen
    float vx = 0;
    float vy = 0;
    float vz = 0;

    for (unsigned int i = 0; i <= (cZResolution); ++i)
    {
        for (unsigned int j = 0; j <= (cAzimResolution); ++j)
        {
            //Vektor-Koordinaten
            vx = cos(deltaAzim * j);
            vy = sin(deltaAzim * j);
            vz = deltaHeight * i;
            cylinderVertices->push_back(osg::Vec3(cRadius * vx, cRadius * vy, vz));
            cylinderNormals->push_back(osg::Vec3(vx, vy, 0));
        }
    }
    cylinderGeometry->setVertexArray(cylinderVertices.get());
    cylinderGeometry->setNormalArray(cylinderNormals.get());
    cylinderGeometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    cylinderGeometry->setColorArray(cylinderColor.get());
    cylinderGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);

    //Vertices mit Primitives versehen
    if (gitter == false)
    {
        //Jewils 4 Vertices zu Primitive "Quad" verbinden
        for (unsigned int i = 0; i < (cZResolution); ++i)
        {
            for (unsigned int j = 0; j < (cAzimResolution); ++j)
            {
                osg::ref_ptr<osg::DrawElementsUInt> cylinderPrim = new osg::DrawElementsUInt(osg::PrimitiveSet::QUADS, 0);
                cylinderPrim->push_back(i * (cAzimResolution + 1) + j);
                cylinderPrim->push_back(i * (cAzimResolution + 1) + j + 1);
                cylinderPrim->push_back((i + 1) * (cAzimResolution + 1) + j + 1);
                cylinderPrim->push_back((i + 1) * (cAzimResolution + 1) + j);

                cylinderGeometry->addPrimitiveSet(cylinderPrim.get());
            }
        }
    }
    else
    {
        //Jewils 4 Vertices über eine Linie "LINE_STRIP" verbinden.
        for (unsigned int i = 0; i < (cZResolution); ++i)
        {
            for (unsigned int j = 0; j < (cAzimResolution); ++j)
            {
                osg::ref_ptr<osg::DrawElementsUInt> cylinderPrim = new osg::DrawElementsUInt(osg::PrimitiveSet::LINE_STRIP, 0);
                cylinderPrim->push_back(i * (cAzimResolution + 1) + j);
                cylinderPrim->push_back(i * (cAzimResolution + 1) + j + 1);
                cylinderPrim->push_back((i + 1) * (cAzimResolution + 1) + j + 1);
                cylinderPrim->push_back((i + 1) * (cAzimResolution + 1) + j);
                cylinderPrim->push_back(i * (cAzimResolution + 1) + j);

                cylinderGeometry->addPrimitiveSet(cylinderPrim.get());
            }
        }
    }

    osg::ref_ptr<osg::Geode> cylinderGeode = new osg::Geode();
    cylinderGeode->addDrawable(cylinderGeometry.get());
    return cylinderGeode.release();
}
