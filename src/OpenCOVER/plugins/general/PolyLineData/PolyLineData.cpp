/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * PolyLineData.cpp
 *
 *  Created on: 30.01.2009
 *      Author: Lukas Pinkowski
 */

#include "PolyLineData.h"

#include <iostream>
#include <fstream>
#include <string>

#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osg/StateSet>
#include <osg/Material>
#include <osg/Geometry>
#include <osg/BoundingBox>
#include <osg/Geode>
#include <osg/Array>
#include <osg/LineWidth>

#include <PluginUtil/PluginMessageTypes.h>

#include <QString>
#include <QStringList>

PolyLineDataPlugin *PolyLineDataPlugin::plugin = 0;

PolyLineDataPlugin::PolyLineDataPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

PolyLineDataPlugin::~PolyLineDataPlugin()
{
}

void PolyLineDataPlugin::tabletEvent(coTUIElement *)
{
}

void PolyLineDataPlugin::tabletPressEvent(coTUIElement *)
{
}

void PolyLineDataPlugin::tabletReleaseEvent(coTUIElement *)
{
}

void PolyLineDataPlugin::preFrame()
{
}

void PolyLineDataPlugin::menuEvent(coMenuItem *)
{
}

void PolyLineDataPlugin::drawInit()
{
}

void PolyLineDataPlugin::message(int toWhom, int type, int len, const void *buf)
{
    switch (type)
    {
    case PluginMessageTypes::WSInterfaceCustomMessage:
    {
        updateData((const char *)buf, len);
        break;
    }

    default:
        break;
    }
}

bool PolyLineDataPlugin::init()
{
    if (plugin)
    {
        return false;
    }
    else
    {
        std::cerr << "PolyLineDataPlugin::init() info: check" << std::endl;
        plugin = this;

        osg::Group *mainGroup = new osg::Group();
        geode = new osg::Geode();

        osg::Matrix m;
        m.makeIdentity();
        m.makeTranslate(osg::Vec3(0.0, 0.0, 0.0));

        osg::MatrixTransform *posTransform = new osg::MatrixTransform;
        posTransform->setMatrix(m);

        this->mtl = new osg::Material;
        this->mtl->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
        this->mtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f,
                                                                       0.9f, 1.0));
        this->mtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f,
                                                                       0.9f, 1.0));
        this->mtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f,
                                                                        0.9f, 1.0));
        this->mtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f,
                                                                        1.0f, 1.0));
        this->mtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);

        osg::LineWidth *lw = new osg::LineWidth;
        lw->setWidth(5.0);

        this->linegeostate = mainGroup->getOrCreateStateSet();
        this->linegeostate->setAttributeAndModes(mtl, osg::StateAttribute::ON);
        this->linegeostate->setAttributeAndModes(lw, osg::StateAttribute::ON);
        this->linegeostate->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
        this->linegeostate->setMode(GL_BLEND, osg::StateAttribute::ON);
        this->linegeostate->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
        this->linegeostate->setNestRenderBins(false);

        //updateData();

        this->fgcolor = new osg::Vec4Array();
        this->fgcolor->push_back(osg::Vec4(1.0, 1.0, 1.0, 1.0));

        // scene graph

        osg::Group *root = cover->getObjectsRoot()->asGroup();
        root->addChild(mainGroup);
        //cover->getScene()->addChild(posTransform);
        mainGroup->addChild(posTransform);
        posTransform->addChild(geode);
    }

    return true;
}

void PolyLineDataPlugin::addGeo()
{

    int position = this->points.size();

    this->points.push_back(new osg::Vec3Array());

    osg::Geometry *lineGeoset = new osg::Geometry();
    lineGeoset->setColorBinding(osg::Geometry::BIND_OVERALL);
    lineGeoset->setColorArray(fgcolor);

    primitives.push_back(new osg::DrawArrays(
        osg::PrimitiveSet::LINE_STRIP, 0, 0));

    lineGeoset->setVertexArray(points[position]);
    lineGeoset->addPrimitiveSet(primitives[position]);

    lineGeoset->setStateSet(linegeostate);

    geode->addDrawable(lineGeoset);

    this->lineGeosets.push_back(lineGeoset);
    this->geode->addDrawable(lineGeoset);
}

//void PolyLineDataPlugin::updateData()
//{
//   std::fstream file;
//   file.open("/home/hpclpink/kieb9/kieb9/ausgabe/p000.dat", fstream::in);
//
//   std::string line;
//   getline(file, line); // skip first two comments...
//   getline(file, line);
//
//   float point = 0.0;
//   float pressure = 0.0;
//   float x = 0.0;
//   float y = 0.0;
//   float z = 0.0;
//
//   const float factor = 1.0f;
//
//   for (int i = 0; i < 64; ++i)
//   {
//      file >> point >> pressure >> x >> y >> z;
//
//      // add to vec3array...
//      std::cerr << "PolyLineDataPlugin::init() info: adding" << x << ", " << y
//            << ", " << z << std::endl;
//      points->push_back(osg::Vec3(factor * -x, factor * y, factor * -z));
//   }
//
//   std::cerr << "PolyLineDataPlugin::init() info: size " << points->size()
//         << std::endl;
//
//   file.close();
//}

void PolyLineDataPlugin::updateData(const char *data, int len)
{

    QString incoming = QString::fromLatin1(data, len);

    if (incoming.startsWith("PolyLine.Data "))
        incoming = incoming.section(' ', 1);
    else
        return;

    int polyLineNumber = incoming.section(' ', 0, 0).toInt();
    incoming = incoming.section(' ', 1);

    while (polyLineNumber >= int(points.size()))
    {
        addGeo();
    }

    points[polyLineNumber]->clear();

    float point = 0.0;
    (void)point;
    float pressure = 0.0;
    (void)pressure;
    float x = 0.0;
    float y = 0.0;
    float z = 0.0;

    QStringList valueList = incoming.split('|');

    std::cerr << "------------------------------------------------" << std::endl;
    for (QStringList::iterator value = valueList.begin(); value != valueList.end(); ++value)
    {

        point = (value++)->toFloat();
        if (value == valueList.end())
            break;
        pressure = (value++)->toFloat();
        if (value == valueList.end())
            break;
        x = (value++)->toFloat();
        if (value == valueList.end())
            break;
        y = (value++)->toFloat();
        if (value == valueList.end())
            break;
        z = value->toFloat();

        std::cerr << "PolyLineDataPlugin::init() info: adding " << x << ", " << y
                  << ", " << z << std::endl;

        points[polyLineNumber]->push_back(osg::Vec3(-x, y, -z));
    }
    std::cerr << "------------------------------------------------" << std::endl;

    primitives[polyLineNumber]->setCount(points[polyLineNumber]->size());
	primitives[polyLineNumber]->dirty();
	points[polyLineNumber]->dirty();
    lineGeosets[polyLineNumber]->dirtyDisplayList();
}

COVERPLUGIN(PolyLineDataPlugin)
