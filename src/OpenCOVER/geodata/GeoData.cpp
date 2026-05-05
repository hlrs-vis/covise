/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "GeoData.h"

#include <cover/coVRPluginSupport.h>

using namespace opencover;

GeoData::GeoData()
{
#ifdef HAVE_PROJ
    m_context = proj_context_create();
#endif

    m_offsetRoot = new osg::PositionAttitudeTransform;
    m_offsetRoot->setPosition(-m_projectOffset);
    cover->getObjectsRoot()->addChild(m_offsetRoot);

    m_terrainRoot = new osg::Group;
    m_offsetRoot->addChild(m_terrainRoot);

    setProjection(GEODATA_DEFAULT_PROJECTION_LOCAL);
}

GeoData::~GeoData()
{
#ifdef HAVE_PROJ
    proj_context_destroy(m_context);

    if (m_transformation)
    {
        proj_destroy(m_transformation);
        m_transformation = nullptr;
    }
#endif
}

GeoData *GeoData::singleton = nullptr;
GeoData *GeoData::instance()
{
    if (!singleton)
        singleton = new GeoData();
    return singleton;
}

void GeoData::setProjection(std::string_view projection)
{
    m_projection = projection;

#ifdef HAVE_PROJ
    if (m_transformation)
    {
        proj_destroy(m_transformation);
    }

    m_transformation = proj_create_crs_to_crs(m_context,
        GEODATA_DEFAULT_PROJECTION_GLOBAL,
        m_projection.c_str(),
        NULL);

    if (m_transformation)
    {
        // Try to normalize the transformation, so that we always use
        // easting-northing = longitude-latitude order, not the order specified
        // in the projection definition (e. g. latitude-longitude for
        // EPSG:4326).
        PJ *tmp = proj_normalize_for_visualization(m_context, m_transformation);
        if (tmp)
        {
            proj_destroy(m_transformation);
            m_transformation = tmp;
        }
    }

#else
    std::cerr << "coVRGeoData: no projection support, geo referencing will not work" << std::endl;
#endif
}

void GeoData::setProjectOffset(const osg::Vec3 &projectOffset)
{
    m_projectOffset = projectOffset;
    m_offsetRoot->setPosition(-m_projectOffset);
}

const std::string &GeoData::projection() const
{
    return m_projection;
}
const osg::Vec3 &GeoData::projectOffset() const
{
    return m_projectOffset;
}

osg::PositionAttitudeTransform *GeoData::offsetRoot()
{
    return m_offsetRoot;
}

osg::Group *GeoData::terrainRoot()
{
    return m_terrainRoot;
}

osg::Vec3 GeoData::toLocal(const osg::Vec3 &globalPosition, bool withOffset) const
{
#ifndef HAVE_PROJ
    return globalPosition;
#else
    PJ_COORD c;
    c.lpz.lam = globalPosition.x();
    c.lpz.phi = globalPosition.y();
    c.lpz.z = globalPosition.z();

    PJ_COORD c_out = proj_trans(m_transformation, PJ_FWD, c);

    osg::Vec3 position_transformed(c_out.enu.e, c_out.enu.n, c_out.enu.u);

    if (withOffset)
        return position_transformed - m_projectOffset;
    else
        return position_transformed;
#endif
}

osg::Vec3 GeoData::toGlobal(const osg::Vec3 &localPosition, bool withOffset) const
{
#ifndef HAVE_PROJ
    return localPosition;
#else
    auto p = withOffset ? localPosition + m_projectOffset : localPosition;

    PJ_COORD c;
    c.enu.e = p.x();
    c.enu.n = p.y();
    c.enu.u = p.z();

    PJ_COORD c_out = proj_trans(m_transformation, PJ_INV, c);

    return osg::Vec3(c_out.lpz.lam, c_out.lpz.phi, c_out.lpz.z);
#endif
}

void GeoData::jumpToLocation(const osg::Vec3 &localPosition)
{
    osg::Vec3 target = localPosition;
    double scale = cover->getScale();

    cover->setXformMat(osg::Matrix::translate(-(localPosition - m_projectOffset) * scale)); // * osg::Matrix::rotate(cover->getXformMat().getRotate()));
}

void GeoData::jumpToLocation(const osg::Vec3 &localPosition, double aboveTerrain)
{
    osg::ref_ptr<osgUtil::LineSegmentIntersector> intersector = new osgUtil::LineSegmentIntersector(
        osg::Vec3(localPosition.x(), localPosition.y(), 10000.0),
        osg::Vec3(localPosition.x(), localPosition.y(), -10000.0));
    osgUtil::IntersectionVisitor iv(intersector);

    auto terrainRoot = GeoData::instance()->terrainRoot();
    if (terrainRoot)
        terrainRoot->accept(iv);

    if (intersector->containsIntersections())
    {
        jumpToLocation(osg::Vec3(
            localPosition.x(),
            localPosition.y(),
            intersector->getFirstIntersection().getLocalIntersectPoint().z() + aboveTerrain));
    }
    else
    {
        jumpToLocation(localPosition);
    }
}
