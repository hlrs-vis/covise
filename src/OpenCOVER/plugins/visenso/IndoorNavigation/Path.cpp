/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Path.h"

#include <cover/VRSceneGraph.h>
#include <cover/coVRPluginSupport.h>
#include <config/CoviseConfig.h>

#include <osg/Shape>
#include <osg/LineWidth>
#include <osg/ShapeDrawable>

#include <stdio.h>
#include <math.h>
using namespace covise;
using namespace opencover;

#define MIN(x, y) (x < y ? x : y)

#define SMOOTHING 50
#define ORIENTATION_SMOOTHING 10

Path::Path()
    : animationIndex(0)
{
    data_lockZ = coCoviseConfig::isOn("COVER.Plugin.IndoorNavigation.LockZ", false);
    data_translation[0] = coCoviseConfig::getFloat("x", "COVER.Plugin.IndoorNavigation.Translation", 0.0f);
    data_translation[1] = coCoviseConfig::getFloat("y", "COVER.Plugin.IndoorNavigation.Translation", 0.0f);
    data_translation[2] = coCoviseConfig::getFloat("z", "COVER.Plugin.IndoorNavigation.Translation", 0.0f);
    data_rotationAxis[0] = coCoviseConfig::getFloat("x", "COVER.Plugin.IndoorNavigation.Rotation", 0.0f);
    data_rotationAxis[1] = coCoviseConfig::getFloat("y", "COVER.Plugin.IndoorNavigation.Rotation", 0.0f);
    data_rotationAxis[2] = coCoviseConfig::getFloat("z", "COVER.Plugin.IndoorNavigation.Rotation", 1.0f);
    data_rotationAngle = coCoviseConfig::getFloat("angle", "COVER.Plugin.IndoorNavigation.Rotation", 0.0f);
    data_scale = coCoviseConfig::getFloat("COVER.Plugin.IndoorNavigation.Scale", 1.0f);
    data_startDrawing = coCoviseConfig::getInt("COVER.Plugin.IndoorNavigation.StartDrawing", 0);

    pathGeode = new osg::Geode();
    pathGeometry = new osg::Geometry();
    pathGeode->addDrawable(pathGeometry.get());
    addChild(pathGeode.get());
}

Path::~Path()
{
    removeChild(pathGeode.get());
}

void Path::update(float animationSeconds)
{
    if (positions.size() == 0)
        return;

    animationIndex = int(animationSeconds);
    animationIndex = MIN(animationIndex, positions.size() - 1);

    drawLine();
}

void Path::updateTransform()
{
    std::vector<osg::Vec3> tmp;

    osg::Matrix m;
    m = m * osg::Matrix::scale(osg::Vec3(data_scale, data_scale, data_scale));
    m = m * osg::Matrix::rotate(data_rotationAngle, data_rotationAxis);
    m = m * osg::Matrix::translate(data_translation);
    std::cerr << "Scale: " << data_scale << " Rotate: " << data_rotationAngle << " Translate: " << data_translation[0] << " " << data_translation[1] << " " << data_translation[2] << std::endl;

    // transform points
    for (int i = 0; i < points.size(); ++i)
    {
        tmp.push_back(points[i]);
        if (data_lockZ)
        {
            tmp[i][2] = tmp[0][2];
        }
        tmp[i] = tmp[i] * m;
    }

    // calculate position smoothing
    positions.clear();
    for (int i = SMOOTHING; i < tmp.size(); ++i)
    {
        osg::Vec3 current;
        for (int j = i - SMOOTHING; j < i; ++j)
        {
            current += tmp[j];
        }
        current /= SMOOTHING;
        positions.push_back(current);
    }

    // calculate orientations
    orientations.clear();
    for (int i = 0; i < ORIENTATION_SMOOTHING; ++i)
    {
        orientations.push_back(osg::Vec3(0.0f, 1.0f, 0.0f));
    }
    for (int i = ORIENTATION_SMOOTHING; i < positions.size(); ++i)
    {
        osg::Vec3 currentO = positions[i] - positions[i - ORIENTATION_SMOOTHING];
        if (currentO.length2() > 0.0001f)
        {
            orientations.push_back(currentO);
        }
        else
        {
            orientations.push_back(orientations[i - 1]);
        }
    }
}

void Path::drawLine()
{
    for (unsigned int i = 0; i < pathGeometry->getNumPrimitiveSets(); i++)
        pathGeometry->removePrimitiveSet(i);
    pathGeometry->setVertexArray(NULL);

    if (data_startDrawing + 1 < animationIndex) // we need at least 2 points to draw
    {
        pathGeometry->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate());
        pathGeometry->getOrCreateStateSet()->setAttributeAndModes(new osg::LineWidth(5), osg::StateAttribute::ON);
        osg::ref_ptr<osg::Vec4Array> pathColor = new osg::Vec4Array();
        pathColor->push_back(osg::Vec4(1.0f, 0.0f, 0.0f, 0.0f));
        pathGeometry->setColorArray(pathColor.get());
        pathGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);
        pathGeometry->getStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

        osg::Vec3Array *vertices = new osg::Vec3Array;
        osg::DrawElementsUInt *conns = new osg::DrawElementsUInt(osg::PrimitiveSet::LINE_STRIP, 0);
        for (int i = data_startDrawing; i < animationIndex; ++i)
        {
            conns->push_back(i - data_startDrawing);
            vertices->push_back(positions[i]);
        }
        pathGeometry->addPrimitiveSet(conns);
        pathGeometry->setVertexArray(vertices);
    }
}

void Path::loadFromFile(std::string filename)
{
    points.clear();
    std::ifstream inFile(filename.c_str());
    std::string line;
    float x, y, z;
    int success;
    while (!inFile.eof())
    {
        inFile >> line;
        success = sscanf(line.c_str(), "P: %G,%G,%G", &x, &y, &z);
        if (success == 3)
        {
            points.push_back(osg::Vec3(x, y, z));
        }
        else
        {
            success = sscanf(line.c_str(), "%G,%G,%G", &x, &y, &z);
            if (success == 3)
                points.push_back(osg::Vec3(x, y, z));
        }
    }
    std::cerr << points.size() << " points read" << std::endl;
    updateTransform();
}

osg::Vec3 Path::getCurrentPosition()
{
    if ((animationIndex >= 0) && (animationIndex < positions.size()))
    {
        return positions[animationIndex];
    }
    else
    {
        return osg::Vec3(0.0f, 0.0f, 0.0f);
    }
}

osg::Vec3 Path::getCurrentOrientation()
{
    if ((animationIndex > 0) && (animationIndex < orientations.size()))
    {
        return orientations[animationIndex];
    }
    else
    {
        return osg::Vec3(0.0f, 1.0f, 0.0f);
    }
}

void Path::changeTranslation(osg::Vec3 offset)
{
    data_translation += offset;
    updateTransform();
}

void Path::changeRotation(float offset)
{
    data_rotationAngle += offset;
    updateTransform();
}

void Path::changeStartDrawing(int offset)
{
    data_startDrawing += offset;
    std::cerr << "StartDrawing: " << data_startDrawing << std::endl;
}
