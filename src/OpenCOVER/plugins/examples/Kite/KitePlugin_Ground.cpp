/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "KitePlugin.h"

#include <cover/coVRPluginSupport.h>

#include <osg/Array>
#include <osg/Depth>
#include <osg/Geode>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/StateSet>
#include <osg/Vec4>

using namespace opencover;

void KitePlugin::createGround()
{
    if (!m_showGround)
        return;

    if (!m_groundXform)
    {
        m_groundXform = new osg::MatrixTransform();
        cover->getObjectsRoot()->addChild(m_groundXform.get());
    }

    if (m_groundGeode)
        return;

    const float s = m_groundSize_m * m_unitsPerMeter * 0.5f;

    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry;
    osg::ref_ptr<osg::Vec3Array> v = new osg::Vec3Array;
    osg::ref_ptr<osg::Vec4Array> c = new osg::Vec4Array;
    osg::ref_ptr<osg::DrawElementsUInt> idx = new osg::DrawElementsUInt(GL_TRIANGLES);

    v->push_back(osg::Vec3(-s, -s, 0.f));
    v->push_back(osg::Vec3( s, -s, 0.f));
    v->push_back(osg::Vec3( s,  s, 0.f));
    v->push_back(osg::Vec3(-s,  s, 0.f));

    idx->push_back(0); idx->push_back(1); idx->push_back(2);
    idx->push_back(0); idx->push_back(2); idx->push_back(3);

    c->push_back(osg::Vec4(0.15f, 0.15f, 0.15f, 0.35f));

    geom->setVertexArray(v.get());
    geom->addPrimitiveSet(idx.get());
    geom->setColorArray(c.get(), osg::Array::BIND_OVERALL);
    osg::StateSet *ss = geom->getOrCreateStateSet();
    ss->setMode(GL_CULL_FACE, osg::StateAttribute::OFF);
    ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    ss->setMode(GL_BLEND, osg::StateAttribute::ON);
    ss->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    osg::ref_ptr<osg::Depth> depth = new osg::Depth;
    depth->setWriteMask(false);
    ss->setAttributeAndModes(depth.get(), osg::StateAttribute::ON);
    ss->setMode(GL_LINE_SMOOTH, osg::StateAttribute::ON);

    m_groundGeode = new osg::Geode;
    m_groundGeode->setName("KitePlugin_Ground");
    m_groundGeode->addDrawable(geom.get());

    // Grid for scale cues.
    osg::ref_ptr<osg::Geometry> grid = new osg::Geometry;
    osg::ref_ptr<osg::Vec3Array> gv = new osg::Vec3Array;
    osg::ref_ptr<osg::Vec4Array> gc = new osg::Vec4Array;
    gc->push_back(osg::Vec4(0.3f, 0.3f, 0.3f, 0.35f));
    grid->setVertexArray(gv.get());
    grid->setColorArray(gc.get(), osg::Array::BIND_OVERALL);

    const float half = s;
    const float step = 20.0f * m_unitsPerMeter;
    for (float x = -half; x <= half; x += step)
    {
        gv->push_back(osg::Vec3(x, -half, 0.f));
        gv->push_back(osg::Vec3(x,  half, 0.f));
    }
    for (float y = -half; y <= half; y += step)
    {
        gv->push_back(osg::Vec3(-half, y, 0.f));
        gv->push_back(osg::Vec3( half, y, 0.f));
    }
    grid->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 0, gv->size()));

    osg::StateSet *gss = grid->getOrCreateStateSet();
    gss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    gss->setMode(GL_BLEND, osg::StateAttribute::ON);
    gss->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

    m_groundGeode->addDrawable(grid.get());

    // attach to ground transform
    m_groundXform->addChild(m_groundGeode.get());

    // Place groundXform at configured ground station position plus optional z-offset.
    const float zW = m_groundZOffset_m * m_unitsPerMeter;
    m_groundPos = osg::Vec3(m_groundPos.x(), m_groundPos.y(), m_groundPos.z() + zW);
    m_groundXform->setMatrix(osg::Matrix::translate(m_groundPos));
}

void KitePlugin::createGroundStation()
{
    if (!m_groundXform)
    {
        m_groundXform = new osg::MatrixTransform();
        cover->getObjectsRoot()->addChild(m_groundXform.get());
        m_groundXform->setMatrix(osg::Matrix::translate(m_groundPos));
    }

    if (m_stationGeode)
        return;

    const float r = 0.5f * m_unitsPerMeter;

    osg::ref_ptr<osg::Sphere> sph = new osg::Sphere(osg::Vec3(0, 0, 0), r);
    osg::ref_ptr<osg::ShapeDrawable> sd = new osg::ShapeDrawable(sph.get());
    sd->setColor(osg::Vec4(0.2f, 0.2f, 0.2f, 1.0f));

    m_stationGeode = new osg::Geode;
    m_stationGeode->setName("KitePlugin_GroundStation");
    m_stationGeode->addDrawable(sd.get());

    // attach to same ground transform
    m_groundXform->addChild(m_stationGeode.get());
}

osg::Vec3 KitePlugin::groundStationWorld() const
{
    // m_groundPos is already in world coords (after our setup)
    return m_groundPos;
}

void KitePlugin::updateGroundAnchor()
{
    if (!m_groundXform)
        return;

    m_groundXform->setMatrix(osg::Matrix::translate(groundStationWorld()));
}

