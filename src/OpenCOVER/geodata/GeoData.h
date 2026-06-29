/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _COVER_GEODATA_GEODATA_H
#define _COVER_GEODATA_GEODATA_H

#include <string>
#include <string_view>

#include <osg/Vec3d>
#include <osg/Matrixd>
#include <osg/PositionAttitudeTransform>

#include <util/coExport.h>

#ifdef HAVE_PROJ
#include <proj.h>
#endif

#define GEODATA_DEFAULT_PROJECTION_GLOBAL "EPSG:4326"
#define GEODATA_DEFAULT_PROJECTION_PROJECT "EPSG:25832"

namespace opencover
{

class COVRGEODATAEXPORT GeoData
{
public:
    GeoData();
    ~GeoData();

    static GeoData *instance();

    void setProjection(std::string_view projection);
    void setProjectOffset(const osg::Vec3d &projectOffset);
    void setProjectTransform(const osg::Vec3d &projectOffset, const double trueNorthDeg);

    const std::string &projection() const;
    const osg::Vec3d &projectOffset() const;
    const double projectTrueNorthDegree() const;
    const osg::Matrixd &projectTransform() const;
    const osg::Matrixd &inverseProjectTransform() const;

    osg::MatrixTransform *transformRoot();
    osg::Group *terrainRoot();

    osg::Vec3d getProjectPosition();
    osg::Vec3d getGlobalPosition();

    /**
     * Transform a global position (easting-northing-altitude) into project
     * (x-y-z) coordinates, applying transform and project offset.
     */
    osg::Vec3d globalToProject(const osg::Vec3d &globalPosition) const;

    /**
     * Transform a global position (easting-northing-altitude) into the
     * reference (x-y-z) coordinates, applying the projection transform, but
     * not the project offset.
     */
    osg::Vec3d globalToReference(const osg::Vec3d &globalPosition) const;

    /**
     * Transform a project (x-y-z) coordinate into a global position
     * (easting-northing-altitude), applying the reverse project offset and
     * transformation.
     */
    osg::Vec3d projectToGlobal(const osg::Vec3d &projectPosition) const;

    void jumpToLocation(const osg::Vec3d &projectPosition);
    void jumpToLocation(const osg::Vec3d &projectPosition, double aboveTerrain);

protected:
#ifdef HAVE_PROJ
    PJ_CONTEXT *m_context = nullptr;
    PJ *m_transformation = nullptr;
#endif

    std::string m_projection;
    osg::Vec3d m_projectOffset;
    double m_projectTrueNorthDegree = 0.0;
    osg::Matrixd m_projectTransform;
    osg::Matrixd m_inverseProjectTransform;

    osg::MatrixTransform *m_transformRoot;
    osg::Group *m_terrainRoot;

private:
    static GeoData *singleton;
};

}
#endif
