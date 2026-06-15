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

    m_transformRoot = new osg::MatrixTransform;
    m_transformRoot->setMatrix(osg::Matrix::identity());
    cover->getObjectsRoot()->addChild(m_transformRoot);

    m_terrainRoot = new osg::Group;
    m_transformRoot->addChild(m_terrainRoot);

    setProjection(GEODATA_DEFAULT_PROJECTION_PROJECT);
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
    setProjectTransform(projectOffset, m_projectTrueNorthDegree);
}

void GeoData::setProjectTransform(const osg::Vec3 &projectOffset, const double trueNorthDeg)
{
    m_projectOffset = projectOffset;
    m_projectTrueNorthDegree = trueNorthDeg;
    osg::Matrix rot = osg::Matrix::rotate(osg::DegreesToRadians(-trueNorthDeg), osg::Vec3(0, 0, 1));
    osg::Matrix trans = osg::Matrix::translate(-projectOffset);
    m_projectTransform = trans * rot;
    m_inverseProjectTransform = osg::Matrix::inverse(m_projectTransform);
    m_transformRoot->setMatrix(m_projectTransform);
}

const std::string &GeoData::projection() const
{
    return m_projection;
}
const osg::Vec3 &GeoData::projectOffset() const
{
    return m_projectOffset;
}
const double GeoData::projectTrueNorthDegree() const
{
    return m_projectTrueNorthDegree;
}
const osg::Matrix &GeoData::projectTransform() const
{
    return m_projectTransform;
}
const osg::Matrix &GeoData::inverseProjectTransform() const
{
    return m_inverseProjectTransform;
}
osg::MatrixTransform *GeoData::transformRoot()
{
    return m_transformRoot;
}

osg::Group *GeoData::terrainRoot()
{
    return m_terrainRoot;
}

osg::Vec3 GeoData::getProjectPosition()
{
    return osg::Matrix::inverse(cover->getXformMat()).getTrans() / cover->getScale();
}

osg::Vec3 GeoData::getGlobalPosition()
{
    auto projectLocation = getProjectPosition();
    return projectToGlobal(projectLocation);
}

osg::Vec3 GeoData::globalToProject(const osg::Vec3 &globalPosition) const
{
    return globalToReference(globalPosition) * m_projectTransform;
}

osg::Vec3 GeoData::globalToReference(const osg::Vec3 &globalPosition) const
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

    return position_transformed;
#endif
}

osg::Vec3 GeoData::projectToGlobal(const osg::Vec3 &projectPosition) const
{
#ifndef HAVE_PROJ
    return projectPosition;
#else
    auto p = projectPosition * m_inverseProjectTransform;

    PJ_COORD c;
    c.enu.e = p.x();
    c.enu.n = p.y();
    c.enu.u = p.z();

    PJ_COORD c_out = proj_trans(m_transformation, PJ_INV, c);

    return osg::Vec3(c_out.lpz.lam, c_out.lpz.phi, c_out.lpz.z);
#endif
}

void GeoData::jumpToLocation(const osg::Vec3 &projectPosition)
{
    osg::Vec3 target = projectPosition;
    double scale = cover->getScale();

    cover->setXformMat(osg::Matrix::translate(-projectPosition * scale)); // * osg::Matrix::rotate(cover->getXformMat().getRotate()));
}

void GeoData::jumpToLocation(const osg::Vec3 &projectPosition, double aboveTerrain)
{
    osg::ref_ptr<osgUtil::LineSegmentIntersector> intersector = new osgUtil::LineSegmentIntersector(
        osg::Vec3(projectPosition.x(), projectPosition.y(), 10000.0),
        osg::Vec3(projectPosition.x(), projectPosition.y(), -10000.0));
    osgUtil::IntersectionVisitor iv(intersector);

    auto terrainRoot = GeoData::instance()->terrainRoot();
    if (terrainRoot)
        terrainRoot->accept(iv);

    if (intersector->containsIntersections())
    {
        jumpToLocation(osg::Vec3(
            projectPosition.x(),
            projectPosition.y(),
            intersector->getFirstIntersection().getLocalIntersectPoint().z() + aboveTerrain));
    }
    else
    {
        jumpToLocation(projectPosition);
    }
}
