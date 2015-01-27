/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <osg/Geometry>
#include <iostream>
#include "coPlotCoordSystem.h"

using std::endl;
using std::cout;

CoordSystem::Graph::Graph(const float *values, size_t nPairs, const osg::Vec4 &color, GLenum renderMode,
                          const std::string &name)
    : m_geometry(0)
    , m_nPairs(nPairs)
    , m_color(0)
    , m_indices(0)
    , m_renderMode(GL_LINE_STRIP)
    , m_values(values)

{
    m_geometry = new osg::Geometry;
    m_geometry->setColorBinding(osg::Geometry::BIND_OVERALL);

    setValues(values, nPairs);
    setRenderMode(renderMode);
    setColor(color);
    setName(name);
    enable();
}

void CoordSystem::Graph::setValues(const float *values, size_t nPairs)
{
    if (!m_geometry)
        return;

    m_nPairs = nPairs;
    osg::ref_ptr<osg::Vec3Array> verts = new osg::Vec3Array();
    for (size_t i = 0; i < m_nPairs * 2; i += 2)
    {
        verts->push_back(osg::Vec3(-values[i], 0.0f, values[i + 1]));
    }
    m_geometry->setVertexArray(verts.get());

    enable();
}

void CoordSystem::Graph::setRenderMode(GLenum newMode)
{
    if (!m_geometry.valid())
        return;
    m_geometry->getPrimitiveSet(0)->setMode(newMode);
    m_renderMode = newMode;
}

void CoordSystem::Graph::setColor(const osg::Vec4 &color)
{
    if (!m_geometry.valid())
        return;

    if (!m_color.valid())
        m_color = new osg::Vec4Array();

    m_color->clear();
    m_color->push_back(color);
    m_geometry->setColorArray(m_color.get());
}

void CoordSystem::Graph::setName(const std::string &name)
{
    m_name = name;
}

void CoordSystem::Graph::enable(void)
{
    if (!m_indices.valid())
        m_indices = new osg::DrawElementsUInt(m_renderMode, 0);
    m_indices->clear();

    for (size_t i = 0; i < m_geometry->getVertexArray()->getNumElements(); i++)
        m_indices->push_back(i);

    m_geometry->addPrimitiveSet(m_indices.get());
}

void CoordSystem::Graph::disable(void)
{
    m_geometry->removePrimitiveSet(0);
}
