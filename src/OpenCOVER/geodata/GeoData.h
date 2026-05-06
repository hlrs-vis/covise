/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _COVER_GEODATA_GEODATA_H
#define _COVER_GEODATA_GEODATA_H

#include <string>
#include <string_view>

#include <osg/Vec3>
#include <osg/Matrix>
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
    void setProjectOffset(const osg::Vec3 &projectOffset);

    const std::string &projection() const;
    const osg::Vec3 &projectOffset() const;

    osg::PositionAttitudeTransform *offsetRoot();
    osg::Group *terrainRoot();

    /**
     * Transform a global position (easting-northing-altitude) into project
     * (x-y-z) coordinates, applying transform and project offset.
     */
    osg::Vec3 globalToProject(const osg::Vec3 &globalPosition) const;

    /**
     * Transform a global position (easting-northing-altitude) into the
     * reference (x-y-z) coordinates, applying the projection transform, but
     * not the project offset.
     */
    osg::Vec3 globalToReference(const osg::Vec3 &globalPosition) const;

    /**
     * Transform a project (x-y-z) coordinate into a global position
     * (easting-northing-altitude), applying the reverse project offset and
     * transformation.
     */
    osg::Vec3 projectToGlobal(const osg::Vec3 &projectPosition) const;

    void jumpToLocation(const osg::Vec3 &projectPosition);
    void jumpToLocation(const osg::Vec3 &projectPosition, double aboveTerrain);

protected:
#ifdef HAVE_PROJ
    PJ_CONTEXT *m_context = nullptr;
    PJ *m_transformation = nullptr;
#endif

    std::string m_projection;
    osg::Vec3 m_projectOffset;

    osg::PositionAttitudeTransform *m_offsetRoot;
    osg::Group *m_terrainRoot;

private:
    static GeoData *singleton;
};

}
#endif
