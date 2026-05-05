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
#define GEODATA_DEFAULT_PROJECTION_LOCAL "EPSG:25832"

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

    /// Transform a global position (easting-northing-altitude) into local
    /// (x-y-z) coordinates, applying transform and project offset.
    osg::Vec3 toLocal(const osg::Vec3 &globalPosition, bool withOffset = true) const;

    /// Transform a local (x-y-z) coordinate into a global position
    /// (easting-northing-altitude), applying the reverse project offset and
    /// transformation.
    osg::Vec3 toGlobal(const osg::Vec3 &localPosition, bool withOffset = true) const;

    void jumpToLocation(const osg::Vec3 &localPosition);
    void jumpToLocation(const osg::Vec3 &localPosition, double aboveTerrain);

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
