/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ScreenPlane.h"
#include "XmlTools.h"

ScreenPlane::ScreenPlane()
    : cHeight(5000.0f)
    , cWidth(5000.0f)
    , cWidthResolution(1)
    , cHeightResolution(1)
{
    geoShape = "Plane";
}

ScreenPlane::ScreenPlane(float width, float height, unsigned int resWidth, unsigned int resHeight)
{
    geoShape = "Plane";
    cHeight = width;
    cWidth = height;
    cWidthResolution = resWidth;
    cHeightResolution = resHeight;
}

ScreenPlane::~ScreenPlane(void)
{
}

void ScreenPlane::saveToXML()
{
    Screen::saveToXML();

    std::string var_str;
    std::string section = "Geometry";
    std::string subsection = geoShape;
    std::string plugPath = XmlTools::getInstance()->getPlugPath();
    std::string path = plugPath + "." + section + "." + subsection;

    XmlTools::getInstance()->saveFloatValue(cHeight, path, "Height");
    XmlTools::getInstance()->saveIntValue(cHeightResolution, path, "HeightRes");
    XmlTools::getInstance()->saveFloatValue(cWidth, path, "Width");
    XmlTools::getInstance()->saveIntValue(cWidthResolution, path, "WidthRes");

    //Einträge in xml schreiben
    XmlTools::getInstance()->saveToXml();
}

void ScreenPlane::loadFromXML()
{
    Screen::loadFromXML();

    std::string section = "Geometry";
    std::string subsection = geoShape;
    std::string plugPath = XmlTools::getInstance()->getPlugPath();
    std::string path = plugPath + "." + section + "." + subsection;

    cHeight = XmlTools::getInstance()->loadFloatValue(path, "Height", 1000.0f);
    cWidth = XmlTools::getInstance()->loadFloatValue(path, "Width", 1000.0f);
    cHeightResolution = XmlTools::getInstance()->loadIntValue(path, "HeightRes", 1);
    cWidthResolution = XmlTools::getInstance()->loadIntValue(path, "WidthRes", 1);
}

void ScreenPlane::setHeight(float height)
{
    cHeight = height;
}

void ScreenPlane::setWidth(float width)
{
    cWidth = width;
}

void ScreenPlane::setHeightResolution(unsigned int resolution)
{
    cHeightResolution = resolution;
}

void ScreenPlane::setWidthResolution(unsigned int resolution)
{
    cWidthResolution = resolution;
}

osg::Geode *ScreenPlane::drawScreen(bool gitter)
{
    float deltaWidth = (cWidth / cWidthResolution);
    float deltaHeight = (cHeight / cHeightResolution);

    osg::ref_ptr<osg::Geometry> quadGeometry = new osg::Geometry();
    osg::ref_ptr<osg::Vec3Array> quadVertices = new osg::Vec3Array;
    osg::ref_ptr<osg::Vec3Array> quadNormals = new osg::Vec3Array;
    osg::ref_ptr<osg::Vec4Array> quadColors = new osg::Vec4Array;
    osg::ref_ptr<osg::Vec2Array> quadTexcoords = new osg::Vec2Array;
    osg::Vec2 dx_texcoord = osg::Vec2(1.0f / (float)(cWidthResolution), 0.0f);
    osg::Vec2 dz_texcoord = osg::Vec2(0.0f, 1.0f / (float)(cHeightResolution));

    //Vertices(=Eckpunkte) über kartesische Koordinaten festlegen
    float vx = 0;
    float vy = 0;
    float vz = 0;

    //Eckpunkte in x-z-Ebene erstellen
    for (unsigned int i = 0; i <= (cHeightResolution); ++i)
    {
        for (unsigned int j = 0; j <= (cWidthResolution); ++j)
        {
            //Eckpunkt-Koordinaten
            vx = (deltaWidth * j) - ((cWidth) / 2.0f);
            vy = 0;
            vz = (deltaHeight * i) - ((cHeight) / 2.0f);
            quadVertices->push_back(osg::Vec3(vx, vy, vz));

            quadNormals->push_back(osg::Vec3(0.0f, -1.0f, 0.0f));
            quadTexcoords->push_back(osg::Vec2(dx_texcoord * j + dz_texcoord * i));
            quadColors->push_back(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
        }
    }
    quadGeometry->setVertexArray(quadVertices.get());
    quadGeometry->setTexCoordArray(0, quadTexcoords.get()); //Jeder Vertex bekommt eine texturkoordinate zugewiesen -> anz. Vertices = anz. Texkoords
    quadGeometry->setColorArray(quadColors.get());
    quadGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);
    quadGeometry->setNormalArray(quadNormals.get());
    quadGeometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

    if (gitter == false)
    {
        //Jewils 4 Vertices zu Primitive "Quad" verbinden
        for (unsigned int i = 0; i < (cHeightResolution); ++i)
        {
            for (unsigned int j = 0; j < (cWidthResolution); ++j)
            {
                osg::ref_ptr<osg::DrawElementsUInt> quadPrim = new osg::DrawElementsUInt(osg::PrimitiveSet::QUADS, 0);
                quadPrim->push_back(i * (cWidthResolution + 1) + j); //li-u
                quadPrim->push_back(i * (cWidthResolution + 1) + j + 1); //li-o
                quadPrim->push_back((i + 1) * (cWidthResolution + 1) + j + 1); //re-o
                quadPrim->push_back((i + 1) * (cWidthResolution + 1) + j); //re-u

                quadGeometry->addPrimitiveSet(quadPrim.get());
            }
        }
    }
    else
    {
        //Jewils 4 Vertices über eine Linie "LINE_STRIP" verbinden.
        for (unsigned int i = 0; i < (cHeightResolution); ++i)
        {
            for (unsigned int j = 0; j < (cWidthResolution); ++j)
            {
                osg::ref_ptr<osg::DrawElementsUInt> quadPrim = new osg::DrawElementsUInt(osg::PrimitiveSet::LINE_STRIP, 0);
                quadPrim->push_back(i * (cWidthResolution + 1) + j); //li-u
                quadPrim->push_back(i * (cWidthResolution + 1) + j + 1); //li-o
                quadPrim->push_back((i + 1) * (cWidthResolution + 1) + j + 1); //re-o
                quadPrim->push_back((i + 1) * (cWidthResolution + 1) + j); //re-u
                quadPrim->push_back(i * (cWidthResolution + 1) + j); //li-u

                quadGeometry->addPrimitiveSet(quadPrim.get());
            }
        }
    }

    osg::ref_ptr<osg::Geode> planeGeode = new osg::Geode();
    planeGeode->addDrawable(quadGeometry.get());
    return planeGeode.release();
}
