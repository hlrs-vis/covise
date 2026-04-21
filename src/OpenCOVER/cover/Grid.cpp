/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Grid.h"

#include <osg/Group>
#include <osg/PolygonOffset>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/StateSet>
#include <osg/LineWidth>
#include <osg/Depth>
#include <osg/ref_ptr>
#include <osgText/Text>
#include <osg/Projection>

namespace opencover
{
osg::ref_ptr<osg::Geode> createGrid(GridProps props)
{
    osg::ref_ptr<osg::Geode> geode = new osg::Geode();

    // Geometry for grid lines
    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry();
    osg::ref_ptr<osg::Vec3Array> verts = new osg::Vec3Array();
    osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array();

    int linesEachSide = static_cast<int>(std::floor(props.halfExtent / props.spacing));
    float start = -linesEachSide * props.spacing;
    float end = linesEachSide * props.spacing;

    // Add X and Z oriented lines parallel to axes (grid on XZ plane, Y up)
    for (int i = -linesEachSide; i <= linesEachSide; ++i)
    {
        float pos = i * props.spacing;

        // Line parallel to X axis at Y = pos
        verts->push_back(osg::Vec3(start, pos, 0));
        verts->push_back(osg::Vec3(end, pos, 0));
        // Color: axis color if pos==0, else gridColor
        if (i == 0)
            colors->push_back(props.axisColors[0]), colors->push_back(props.axisColors[0]);
        else
            colors->push_back(props.gridColor), colors->push_back(props.gridColor);

        // Line parallel to Y axis at X = pos
        verts->push_back(osg::Vec3(pos, start, 0));
        verts->push_back(osg::Vec3(pos, end, 0));
        if (i == 0)
            colors->push_back(props.axisColors[1]), colors->push_back(props.axisColors[1]);
        else
            colors->push_back(props.gridColor), colors->push_back(props.gridColor);
    }

    geom->setVertexArray(verts.get());
    geom->setColorArray(colors.get(), osg::Array::BIND_PER_VERTEX);
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    // Primitive set: pair of vertices per line => GL_LINES
    geom->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 0, verts->size()));

    // StateSet: line width, depth test, optionally render on top
    osg::ref_ptr<osg::StateSet> ss = geom->getOrCreateStateSet();
    osg::ref_ptr<osg::LineWidth> lw = new osg::LineWidth(props.thinLineWidth);
    ss->setAttributeAndModes(lw, osg::StateAttribute::ON);
    ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    // If renderOnTop: disable depth write and use a render bin
    if (props.renderOnTop)
    {
        ss->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF);
        ss->setRenderBinDetails(11, "RenderBin");
    }
    else
    {
        ss->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);
    }

    // Slight polygon offset to reduce z-fighting if necessary
    ss->setAttributeAndModes(new osg::PolygonOffset(-1.0f, -1.0f), osg::StateAttribute::ON);

    // Add geometry to geode
    geode->addDrawable(geom.get());

    // Add thicker axis lines if desired (drawn as separate Geometry so line width differs)
    osg::ref_ptr<osg::Geometry> axesGeom = new osg::Geometry();
    osg::ref_ptr<osg::Vec3Array> aVerts = new osg::Vec3Array();
    osg::ref_ptr<osg::Vec4Array> aColors = new osg::Vec4Array();

    // X axis
    aVerts->push_back(osg::Vec3(-props.halfExtent, 0, 0));
    aVerts->push_back(osg::Vec3(props.halfExtent, 0, 0));
    aColors->push_back(props.axisColors[0]);
    aColors->push_back(props.axisColors[0]);

    // Y axis
    aVerts->push_back(osg::Vec3(0, -props.halfExtent, 0));
    aVerts->push_back(osg::Vec3(0, props.halfExtent, 0));
    aColors->push_back(props.axisColors[1]);
    aColors->push_back(props.axisColors[1]);

    // Z axis
    aVerts->push_back(osg::Vec3(0, 0, -props.halfExtent));
    aVerts->push_back(osg::Vec3(0, 0, props.halfExtent));
    aColors->push_back(props.axisColors[2]);
    aColors->push_back(props.axisColors[2]);

    axesGeom->setVertexArray(aVerts.get());
    axesGeom->setColorArray(aColors.get(), osg::Array::BIND_PER_VERTEX);
    axesGeom->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 0, aVerts->size()));

    osg::ref_ptr<osg::StateSet> axesSS = axesGeom->getOrCreateStateSet();
    osg::ref_ptr<osg::LineWidth> axesLW = new osg::LineWidth(props.thickLineWidth);
    axesSS->setAttributeAndModes(axesLW, osg::StateAttribute::ON);
    axesSS->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    if (props.renderOnTop)
    {
        axesSS->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF);
        axesSS->setRenderBinDetails(12, "RenderBin");
    }
    else
    {
        axesSS->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);
    }

    axesGeom->getOrCreateStateSet()->setAttributeAndModes(new osg::PolygonOffset(-1.0f, -1.0f), osg::StateAttribute::ON);
    geode->addDrawable(axesGeom.get());

    return geode;
}
}
