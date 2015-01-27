/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ScreenDome.h"
#include "XmlTools.h"
#include <osgUtil/SmoothingVisitor>

ScreenDome::ScreenDome()
    : cRadius(2500.0f)
    , cAzimResolution(100)
    , cPolarResolution(100)
    , cAzimSegSize(180.0f)
    , cPolarSegSize(180.0f)
{
    geoShape = "Dome";
}

ScreenDome::~ScreenDome(void)
{
}

void ScreenDome::saveToXML()
{
    Screen::saveToXML();

    std::string var_str;
    std::string section = "Geometry";
    std::string subsection = geoShape;
    std::string plugPath = XmlTools::getInstance()->getPlugPath();
    std::string path = plugPath + "." + section + "." + subsection;

    XmlTools::getInstance()->saveFloatValue(cRadius, path, "Radius");
    XmlTools::getInstance()->saveIntValue(cAzimSegSize, path, "AzimAngle");
    XmlTools::getInstance()->saveFloatValue(cPolarSegSize, path, "PolarAngle");
    XmlTools::getInstance()->saveIntValue(cAzimResolution, path, "AzimRes");
    XmlTools::getInstance()->saveFloatValue(cPolarResolution, path, "PolarRes");

    //Einträge in xml schreiben
    XmlTools::getInstance()->saveToXml();
}

void ScreenDome::loadFromXML()
{
    Screen::loadFromXML();

    std::string section = "Geometry";
    std::string subsection = geoShape;
    std::string plugPath = XmlTools::getInstance()->getPlugPath();
    std::string path = plugPath + "." + section + "." + subsection;

    cRadius = XmlTools::getInstance()->loadFloatValue(path, "Radius", 2500.0f);
    cAzimSegSize = XmlTools::getInstance()->loadFloatValue(path, "AzimAngle", 180.0f);
    cPolarSegSize = XmlTools::getInstance()->loadFloatValue(path, "PolarAngle", 180.0f);
    cAzimResolution = XmlTools::getInstance()->loadIntValue(path, "AzimRes", 100);
    cPolarResolution = XmlTools::getInstance()->loadIntValue(path, "PolarRes", 100);
}

void ScreenDome::setRadius(float radius)
{
    cRadius = radius;
}

void ScreenDome::setAzimSegmentSize(float segSize)
{
    cAzimSegSize = segSize;
}

void ScreenDome::setPolarSegmentSize(float segSize)
{
    cPolarSegSize = segSize;
}

void ScreenDome::setAzimResolution(unsigned int resolution)
{
    cAzimResolution = resolution;
}

void ScreenDome::setPolarResolution(unsigned int resolution)
{
    cPolarResolution = resolution;
}

osg::Geode *ScreenDome::drawScreen(bool gitter)
{
    float deltaAzim = (2 * osg::PI * ((cAzimSegSize) / 360)) / (cAzimResolution);
    float deltaPolar = (osg::PI * ((cPolarSegSize) / 180)) / (cPolarResolution);

    osg::ref_ptr<osg::Geometry> domeGeometry = new osg::Geometry();
    osg::ref_ptr<osg::Vec3Array> domeVertices = new osg::Vec3Array;
    osg::ref_ptr<osg::Vec3Array> domeNormals = new osg::Vec3Array;
    osg::ref_ptr<osg::Vec4Array> domeColor = new osg::Vec4Array;
    domeColor->push_back(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));

    //Vertices über Polarkoordinaten festlegen
    float vx = 0;
    float vy = 0;
    float vz = 0;

    for (unsigned int i = 0; i <= (cPolarResolution); ++i)
    {
        for (unsigned int j = 0; j <= (cAzimResolution); ++j)
        {
            //Vektor-Koordinaten
            vx = sin(deltaPolar * i) * cos(deltaAzim * j);
            vy = sin(deltaPolar * i) * sin(deltaAzim * j);
            vz = cos(deltaPolar * i);
            domeVertices->push_back(osg::Vec3(cRadius * vx, cRadius * vy, cRadius * vz));
            //domeNormals->push_back( osg::Vec3(vx,vy,vz) );
        }
    }
    domeGeometry->setVertexArray(domeVertices.get());
    //domeGeometry->setNormalArray( domeNormals.get() );
    //domeGeometry->setNormalBinding( osg::Geometry::BIND_PER_PRIMITIVE );
    domeGeometry->setColorArray(domeColor.get());
    domeGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);
    if (gitter == false)
    {
        //Jewils 4 Vertices zu Primitive "Quad" verbinden
        for (unsigned int i = 0; i < (cPolarResolution); ++i)
        {
            for (unsigned int j = 0; j < (cAzimResolution); ++j)
            {
                osg::ref_ptr<osg::DrawElementsUInt> domePrim = new osg::DrawElementsUInt(osg::PrimitiveSet::QUADS, 0);
                domePrim->push_back(i * (cAzimResolution + 1) + j);
                domePrim->push_back(i * (cAzimResolution + 1) + j + 1);
                domePrim->push_back((i + 1) * (cAzimResolution + 1) + j + 1);
                domePrim->push_back((i + 1) * (cAzimResolution + 1) + j);

                domeGeometry->addPrimitiveSet(domePrim.get());
            }
        }
    }
    else
    {
        //Jewils 4 Vertices über eine Linie "LINE_STRIP" verbinden.
        for (unsigned int i = 0; i < (cPolarResolution); ++i)
        {
            for (unsigned int j = 0; j < (cAzimResolution); ++j)
            {
                osg::ref_ptr<osg::DrawElementsUInt> domePrim = new osg::DrawElementsUInt(osg::PrimitiveSet::LINE_STRIP, 0);
                domePrim->push_back(i * (cAzimResolution + 1) + j);
                domePrim->push_back(i * (cAzimResolution + 1) + j + 1);
                domePrim->push_back((i + 1) * (cAzimResolution + 1) + j + 1);
                domePrim->push_back((i + 1) * (cAzimResolution + 1) + j);
                domePrim->push_back(i * (cAzimResolution + 1) + j);

                domeGeometry->addPrimitiveSet(domePrim.get());
            }
        }
    }

    //normalen automatisch berechnen
    osgUtil::SmoothingVisitor::smooth(*(domeGeometry.get()));

    osg::ref_ptr<osg::Geode> domeGeode = new osg::Geode();
    domeGeode->addDrawable(domeGeometry.get());
    return domeGeode.release();
}